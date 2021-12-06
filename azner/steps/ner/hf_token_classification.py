import logging
import traceback
from typing import List, Tuple, Dict

import torch
from pytorch_lightning import Trainer
from torch import Tensor
from torch.nn import Softmax
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    BatchEncoding,
)
from transformers.file_utils import PaddingStrategy

from azner.data.data import Document, Section, NerProcessedSection, PROCESSING_EXCEPTION
from azner.data.pytorch import HFDataset
from azner.modelling.hf_lightning_wrappers import PLAutoModelForTokenClassification
from azner.steps import BaseStep
from azner.steps.ner.bio_label_parser import BIOLabelParser
from azner.utils.utils import documents_to_document_section_batch_encodings_map
from azner.steps.ner.bio_label_preprocessor import BioLabelPreProcessor

logger = logging.getLogger(__name__)


class TransformersModelForTokenClassificationNerStep(BaseStep):
    """
    An wrapper for :class:`transformers.AutoModelForTokenClassification' and
    :class:`azner.steps.ner.bio_label_preprocessor.BioLabelPreProcessor`. This implementation uses a sliding
    window concept to process large documents that don't fit into the maximum sequence length allowed by a model.

    """

    def __init__(
        self,
        path: str,
        depends_on: List[str],
        batch_size: int,
        stride: int,
        max_sequence_length: int,
        trainer: Trainer,
        debug=False,
    ):
        """
        :param stride: passed to HF tokenizers (for splitting long docs)
        :param max_sequence_length: passed to HF tokenizers (for splitting long docs)
        :param path: path to HF model, config and tokenizer. Passed to HF .from_pretrained()
        :param depends_on:
        :param batch_size: batch size for dataloader
        :param debug: print extra logging info
        """
        super().__init__(depends_on=depends_on)
        self.debug = debug
        if max_sequence_length % 2 != 0:
            raise RuntimeError(
                "max_sequence_length must %2 ==0 in order for correct document windowing"
            )
        self.max_sequence_length = max_sequence_length
        if stride % 2 != 0:
            raise RuntimeError("stride must %2 ==0 in order for correct document windowing")
        self.stride = stride
        self.batch_size = batch_size
        self.config = AutoConfig.from_pretrained(path)
        self.tokeniser = AutoTokenizer.from_pretrained(path, config=self.config)
        self.model = AutoModelForTokenClassification.from_pretrained(path, config=self.config)
        self.model = PLAutoModelForTokenClassification(self.model)
        self.trainer = trainer
        self.softmax = Softmax(dim=-1)
        self.entity_mapper = BIOLabelParser(
            list(self.config.id2label.values()), namespace=self.namespace()
        )
        self.bio_preprocessor = BioLabelPreProcessor()

    def get_dataloader(self, docs: List[Document]) -> Tuple[DataLoader, Dict[int, Section]]:
        """
        get a dataloader from a List of Document. Collation is handled via DataCollatorWithPadding

        :param docs:
        :return: a tuple of dataloader, and a dict of int:Section. The int maps to overflow_to_sample_mapping in the
                underlying batch encoding, allowing the processing of docs longer than can fit within the maximum
                sequence length of a transformer
        """
        batch_encoding, id_section_map = documents_to_document_section_batch_encodings_map(
            docs, self.tokeniser, stride=self.stride, max_length=self.max_sequence_length
        )

        if self.debug:
            decoded_texts = [
                self.tokeniser.decode(encoding.ids) for encoding in batch_encoding.encodings
            ]
            for decoded_text in decoded_texts:
                logger.info(f"inputs to model this call: {decoded_text}")

        dataset = HFDataset(batch_encoding)
        collate_func = DataCollatorWithPadding(
            tokenizer=self.tokeniser, padding=PaddingStrategy.MAX_LENGTH
        )
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=collate_func)
        return loader, id_section_map

    def merge_section_frames(
        self,
        section_index: int,
        batch_encoding: BatchEncoding,
        confidence_and_labels_tensor: Tuple[Tensor, Tensor],
    ) -> NerProcessedSection:
        """
        for a given section index, obtain a NerProcessedSection representing all of the inferred labels for that section
        :param section_index: int of the section index
        :param batch_encoding: the BatchEncoding for this dataset
        :param confidence_and_labels_tensor: a tuple of the confidence and labels from the model
        :return:
        """
        all_frame_offsets = []
        all_frame_word_ids = []
        all_frame_labels = []
        all_frame_confidences = []

        section_frame_indices = self.get_list_of_batch_encoding_frames_for_section(
            batch_encoding, section_index
        )
        for frame_index, section_frame_index in enumerate(section_frame_indices):
            (
                frame_offsets,
                frame_word_ids,
                frame_labels,
                frame_confidence,
            ) = self.get_offset_and_word_id_frames(
                batch_encoding=batch_encoding,
                number_of_frames=len(section_frame_indices),
                frame_index=frame_index,
                section_frame_index=section_frame_index,
                confidence_and_labels_tensor=confidence_and_labels_tensor,
            )
            all_frame_confidences.extend(frame_confidence.numpy().tolist())
            all_frame_labels.extend(frame_labels.numpy().tolist())
            all_frame_word_ids.extend(frame_word_ids)
            all_frame_offsets.extend(frame_offsets)

        return NerProcessedSection(
            all_frame_offsets=all_frame_offsets,
            all_frame_labels=all_frame_labels,
            all_frame_word_ids=all_frame_word_ids,
            all_frame_confidences=all_frame_confidences,
        )

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        try:
            loader, id_section_map = self.get_dataloader(docs)
            # run the transformer and get results
            confidence_and_labels_tensor = self.get_confidence_and_labels_tensor(loader)
            for section_index, section in id_section_map.items():
                # for long docs, we need to split section.get_text() into frames (i.e. portions that will fit into Bert or
                # similar)
                ner_processed_section = self.merge_section_frames(
                    section_index=section_index,
                    batch_encoding=loader.dataset.encodings,
                    confidence_and_labels_tensor=confidence_and_labels_tensor,
                )
                all_words = ner_processed_section.to_tokenized_words(self.config.id2label)
                transformed_words = self.bio_preprocessor(all_words)
                for transformed_word in transformed_words:
                    for i, label in enumerate(transformed_word.word_labels_strings):
                        if self.debug:
                            logger.info(
                                f"processing label: {label} for token {section.get_text()[transformed_word.word_offsets[i][0]:transformed_word.word_offsets[i][1]]}"
                            )
                        self.entity_mapper.update_parse_states(
                            label,
                            offsets=transformed_word.word_offsets[i],
                            text=section.get_text(),
                            confidence=transformed_word.word_confidences[i],
                        )

                # at the end of the section, get the results
                section.entities = self.entity_mapper.get_entities()
                # reset the entity mapper in preparation for the next section
                self.entity_mapper.reset()
        except Exception:
            affected_doc_ids = [doc.idx for doc in docs]
            for doc in docs:
                message = (
                    f"batch failed: affected ids: {affected_doc_ids}\n" + traceback.format_exc()
                )
                doc.metadata[PROCESSING_EXCEPTION] = message
                failed_docs.append(doc)

        return docs, failed_docs

    def get_list_of_batch_encoding_frames_for_section(
        self, batch_encoding: BatchEncoding, section_index: int
    ) -> List[int]:
        """
        for a given dataloader with a HFDataset, return a list of frame indexes associated with a given section index
        :param loader:
        :param section_index:
        :return:
        """
        section_frame_indices = [
            i
            for i, x in enumerate(batch_encoding.data["overflow_to_sample_mapping"])
            if x == section_index
        ]
        return section_frame_indices

    def get_confidence_and_labels_tensor(self, loader: DataLoader) -> Tuple[Tensor, Tensor]:
        """
        get a namedtuple_values_indices consisting of confidence and labels for a given dataloader (i.e. run bert)
        :param loader:
        :return:
        """
        results = torch.cat(
            [
                x.logits
                for x in self.trainer.predict(
                    model=self.model, dataloaders=loader, return_predictions=True
                )
            ]
        )
        softmax = self.softmax(results)
        # get confidence scores and label ints
        confidence_and_labels_tensor = torch.max(softmax, dim=-1)
        return confidence_and_labels_tensor

    def get_offset_and_word_id_frames(
        self,
        batch_encoding: BatchEncoding,
        number_of_frames: int,
        frame_index: int,
        section_frame_index: int,
        confidence_and_labels_tensor: Tuple[Tensor, Tensor],
    ) -> Tuple[List[Tuple[int, int]], List[int], Tensor, Tensor]:
        """
        depending on the number of frames generated by a string of text, and whether it is the first or last frame,
        we need to return different subsets of the frame offsets and frame word_ids
        :param batch_encoding: a HF BatchEncoding
        :param number_of_frames: number of frames created by the tokenizer for the string
        :param frame_index: the index of the query frame, relative to the total number of frames
        :param section_frame_index: the index of the section frame, relative to the whole BatchEncoding
        :return: Tuple of 2 lists: frame offsets and frame word ids
        """
        half_stride = int(self.stride / 2)
        # 1:-1 skip cls and sep tokens
        if number_of_frames == 1:
            start_index = 1
            end_index = -1
        elif number_of_frames > 1 and frame_index == 0:
            start_index = 1
            end_index = -(half_stride + 1)
        elif number_of_frames > 1 and frame_index == number_of_frames - 1:
            start_index = half_stride + 1
            end_index = -1
        else:
            start_index = half_stride + 1
            end_index = -(half_stride + 1)

        frame_offsets = batch_encoding.encodings[section_frame_index].offsets[start_index:end_index]
        frame_word_ids = batch_encoding.encodings[section_frame_index].word_ids[
            start_index:end_index
        ]
        frame_input_ids = batch_encoding.encodings[section_frame_index].ids
        frame_labels = confidence_and_labels_tensor[1][section_frame_index][start_index:end_index]
        frame_confidence = confidence_and_labels_tensor[0][section_frame_index][
            start_index:end_index
        ]

        if self.debug:
            logger.info(
                f"inputs this frame: {self.tokeniser.decode(frame_input_ids[:start_index])}<-IGNORED "
                f"START->{self.tokeniser.decode(frame_input_ids[start_index:end_index])}"
                f"<-END IGNORED->{self.tokeniser.decode(frame_input_ids[end_index:])}"
            )
        return frame_offsets, frame_word_ids, frame_labels, frame_confidence
