import logging
from collections.abc import Iterator
from typing import Any, Iterable, Optional, cast

import torch
from torch import Tensor, softmax
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)
from transformers.file_utils import PaddingStrategy

from kazu.data import Document, Section
from kazu.steps import Step, document_batch_step
from kazu.steps.ner.entity_post_processing import NonContiguousEntitySplitter
from kazu.steps.ner.tokenized_word_processor import (
    TokenizedWord,
    TokenizedWordProcessor,
)
from kazu.utils.utils import documents_to_document_section_batch_encodings_map

logger = logging.getLogger(__name__)


class HFDataset(IterableDataset[dict[str, Any]]):
    def __getitem__(self, index: int) -> dict[str, Any]:
        return {key: self.encodings.data[key][index] for key in self.keys_to_use}

    def __init__(self, encodings: BatchEncoding, keys_to_use: Iterable[str]):
        """Simple implementation of :class:`torch.utils.data.IterableDataset`\\ ,
        producing HF tokenizer input_id.

        :param encodings:
        :param keys_to_use: the keys to use from the encodings (not all models require
            token_type_ids)
        """
        self.keys_to_use = set(keys_to_use)
        self.encodings = encodings
        self.dataset_size = len(encodings.data["input_ids"])

    def __iter__(self) -> Iterator[dict[str, Any]]:

        for i in range(self.dataset_size):
            yield {key: self.encodings.data[key][i] for key in self.keys_to_use}


class TransformersModelForTokenClassificationNerStep(Step):
    """A wrapper for :class:`transformers.AutoModelForTokenClassification`\\ .

    This implementation uses a sliding
    window concept to process large documents that don't fit into the maximum sequence length
    allowed by a model. Resulting token labels are then post processed by
    :class:`~kazu.steps.ner.tokenized_word_processor.TokenizedWordProcessor`.
    """

    def __init__(
        self,
        path: str,
        batch_size: int,
        stride: int,
        max_sequence_length: int,
        tokenized_word_processor: TokenizedWordProcessor,
        keys_to_use: Iterable[str],
        entity_splitter: Optional[NonContiguousEntitySplitter] = None,
        device: str = "cpu",
    ):
        """

        :param path: path to HF model, config and tokenizer. Passed to HF .from_pretrained()
        :param batch_size: batch size for dataloader
        :param stride: passed to HF tokenizers (for splitting long docs)
        :param max_sequence_length: passed to HF tokenizers (for splitting long docs)
        :param tokenized_word_processor:
        :param keys_to_use: keys to use from the encodings. Note that this varies depending on the flaour of bert model (e.g. distilbert requires token_type_ids)
        :param entity_splitter: to detect non-contiguous entities if provided
        :param device: device to run the model on. Defaults to "cpu"
        """

        self.keys_to_use = set(keys_to_use)
        self.device = device
        self.entity_splitter = entity_splitter
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
        self.tokeniser: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            path, config=self.config
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            path, config=self.config
        ).eval()
        self.tokenized_word_processor = tokenized_word_processor
        self.model.to(device)

    @document_batch_step
    def __call__(self, docs: list[Document]) -> None:
        loader, id_section_map = self.get_dataloader(docs)
        # need this so mypy knows to expect the dataset to have encodings
        dataset = cast(HFDataset, loader.dataset)
        # run the transformer and get results
        if self.tokenized_word_processor.use_multilabel:
            activations = self.get_multilabel_activations(loader)
        else:
            activations = self.get_single_label_activations(loader)
        for section_index, section in id_section_map.items():
            words = self.section_frames_to_tokenised_words(
                section_index=section_index,
                batch_encoding=dataset.encodings,
                predictions=activations,
            )
            entities = self.tokenized_word_processor(
                words, text=section.text, namespace=self.namespace()
            )
            section.entities.extend(entities)
            if self.entity_splitter:
                for ent in entities:
                    section.entities.extend(self.entity_splitter(ent, section.text))

    def get_dataloader(self, docs: list[Document]) -> tuple[DataLoader, dict[int, Section]]:
        """Get a dataloader from a List of :class:`kazu.data.Document`. Collation is
        handled via :class:`transformers.DataCollatorWithPadding`\\ .

        :param docs:
        :return: The returned dict's keys map to overflow_to_sample_mapping in the
            underlying batch encoding, allowing the processing of docs longer than can
            fit within the maximum sequence length of a transformer
        """
        batch_encoding, id_section_map = documents_to_document_section_batch_encodings_map(
            docs, self.tokeniser, stride=self.stride, max_length=self.max_sequence_length
        )
        dataset = HFDataset(batch_encoding, keys_to_use=self.keys_to_use)
        collate_func = DataCollatorWithPadding(
            tokenizer=self.tokeniser, padding=PaddingStrategy.MAX_LENGTH
        )
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=collate_func)
        return loader, id_section_map

    def frame_to_tok_word(
        self,
        batch_encoding: BatchEncoding,
        number_of_frames: int,
        frame_index: int,
        section_frame_index: int,
        predictions: Tensor,
    ) -> list[TokenizedWord]:
        """Depending on the number of frames generated by a string of text, and whether
        it is the first or last frame, we need to return different subsets of the frame
        offsets and frame word_ids.

        :param batch_encoding:
        :param number_of_frames: number of frames created by the tokenizer for the
            string
        :param frame_index: the index of the query frame, relative to the total number
            of frames
        :param section_frame_index: the index of the section frame, relative to the
            whole BatchEncoding
        :param predictions:
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

        assert batch_encoding.encodings is not None
        frame_offsets = batch_encoding.encodings[section_frame_index].offsets[start_index:end_index]
        frame_word_ids: list[Optional[int]] = batch_encoding.encodings[
            section_frame_index
        ].word_ids[start_index:end_index]
        frame_token_ids = batch_encoding.encodings[section_frame_index].ids[start_index:end_index]
        frame_tokens = batch_encoding.encodings[section_frame_index].tokens[start_index:end_index]
        predictions = predictions[section_frame_index][start_index:end_index]
        prev_word_id: Optional[int] = None
        all_words = []

        word_id_index_start, offset_start, offset_end = 0, 0, 0

        for i, word_id in enumerate(frame_word_ids):
            if word_id != prev_word_id:
                # start a new word, and add the previous
                if prev_word_id is not None:
                    all_words.append(
                        TokenizedWord(
                            token_offsets=frame_offsets[word_id_index_start:i],
                            token_confidences=predictions[word_id_index_start:i],
                            token_ids=frame_token_ids[word_id_index_start:i],
                            tokens=frame_tokens[word_id_index_start:i],
                            word_char_start=offset_start,
                            word_char_end=offset_end - 1,
                            word_id=prev_word_id,
                        )
                    )
                word_id_index_start = i
                offset_start, offset_end = frame_offsets[i]
            if i == len(frame_word_ids) - 1 and word_id is not None:
                # if checking the last word in a frame, if word_id is not None, add it
                all_words.append(
                    TokenizedWord(
                        token_offsets=frame_offsets[word_id_index_start : i + 1],
                        token_confidences=predictions[word_id_index_start : i + 1],
                        token_ids=frame_token_ids[word_id_index_start : i + 1],
                        tokens=frame_tokens[word_id_index_start : i + 1],
                        word_char_start=offset_start,
                        word_char_end=offset_end,
                        word_id=word_id,
                    )
                )

            _, offset_end = frame_offsets[i]
            prev_word_id = word_id

        frame_word_ids_excluding_nones = set(frame_word_ids)
        frame_word_ids_excluding_nones.discard(None)
        assert len(set(frame_word_ids_excluding_nones)) == len(all_words)

        logger.debug(
            f"inputs this frame: {self.tokeniser.decode(frame_token_ids[:start_index])}<-IGNORED "
            f"START->{self.tokeniser.decode(frame_token_ids[start_index:end_index])}"
            f"<-END IGNORED->{self.tokeniser.decode(frame_token_ids[end_index:])}"
        )
        return all_words

    def section_frames_to_tokenised_words(
        self,
        section_index: int,
        batch_encoding: BatchEncoding,
        predictions: Tensor,
    ) -> list[TokenizedWord]:
        """

        :param section_index:
        :param batch_encoding:
        :param predictions:
        :return:
        """
        words = []

        section_frame_indices = self.get_list_of_batch_encoding_frames_for_section(
            batch_encoding, section_index
        )
        for frame_index, section_frame_index in enumerate(section_frame_indices):
            word_sub_list = self.frame_to_tok_word(
                batch_encoding=batch_encoding,
                number_of_frames=len(section_frame_indices),
                frame_index=frame_index,
                section_frame_index=section_frame_index,
                predictions=predictions,
            )
            words.extend(word_sub_list)
        return words

    @staticmethod
    def get_list_of_batch_encoding_frames_for_section(
        batch_encoding: BatchEncoding, section_index: int
    ) -> list[int]:
        """For a given dataloader with a HFDataset, return a list of frame indexes
        associated with a given section index.

        :param batch_encoding:
        :param section_index:
        :return:
        """
        section_frame_indices = [
            i
            for i, x in enumerate(batch_encoding.data["overflow_to_sample_mapping"])
            if x == section_index
        ]
        return section_frame_indices

    def get_multilabel_activations(self, loader: DataLoader) -> Tensor:
        """Get a tensor consisting of confidences for labels in a multi label
        classification context. Output tensor is of shape (n_samples,
        max_sequence_length, n_labels).

        :param loader:
        :return:
        """
        with torch.no_grad():
            results = torch.cat(
                tuple(self.model(**batch.to(self.device)).logits.to("cpu") for batch in loader)
            ).to(self.device)
        return results.heaviside(torch.tensor([0.0]).to(self.device)).int().to("cpu")

    def get_single_label_activations(self, loader: DataLoader) -> Tensor:
        """Get a tensor consisting of one hot binary classifications in a single label
        classification context. Output tensor is of shape (n_samples,
        max_sequence_length, n_labels).

        :param loader:
        :return:
        """
        with torch.no_grad():
            results = torch.cat(
                tuple(self.model(**batch.to(self.device)).logits.to("cpu") for batch in loader)
            )
            return softmax(results, dim=-1).to("cpu")
