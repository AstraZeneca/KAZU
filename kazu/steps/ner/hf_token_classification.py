import logging
from collections.abc import Iterable, Iterator, Callable
from functools import partial
from typing import Optional, cast, Any

import torch
from torch import Tensor, sigmoid, softmax
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoModelForTokenClassification,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    BatchEncoding,
    PreTrainedTokenizerBase,
)
from transformers.file_utils import PaddingStrategy

from kazu.data import Document, Section
from kazu.steps import Step, document_batch_step
from kazu.steps.ner.entity_post_processing import NonContiguousEntitySplitter
from kazu.steps.ner.tokenized_word_processor import TokenizedWordProcessor, TokenizedWord
from kazu.utils.utils import documents_to_document_section_batch_encodings_map

logger = logging.getLogger(__name__)


class HFDataset(IterableDataset[dict[str, Any]]):
    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "input_ids": self.encodings.data["input_ids"][index],
            "attention_mask": self.encodings.data["attention_mask"][index],
            "token_type_ids": self.encodings.data["token_type_ids"][index],
        }

    def __init__(self, encodings: BatchEncoding):
        """Simple implementation of :class:`torch.utils.data.IterableDataset`\\ ,
        producing HF tokenizer input_id.

        :param encodings:
        """
        self.encodings = encodings
        self.dataset_size = len(encodings.data["input_ids"])

    def __iter__(self) -> Iterator[dict[str, Any]]:

        for i in range(self.dataset_size):
            yield {
                "input_ids": self.encodings.data["input_ids"][i],
                "attention_mask": self.encodings.data["attention_mask"][i],
                "token_type_ids": self.encodings.data["token_type_ids"][i],
            }


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
        labels: list[str],
        detect_subspans: bool = False,
        threshold: Optional[float] = None,
        entity_splitter: Optional[NonContiguousEntitySplitter] = None,
        strip_re: Optional[dict[str, str]] = None,
    ):
        """

        :param path: path to HF model, config and tokenizer. Passed to HF .from_pretrained()
        :param batch_size: batch size for dataloader
        :param stride: passed to HF tokenizers (for splitting long docs)
        :param max_sequence_length: passed to HF tokenizers (for splitting long docs)
        :param labels:
        :param detect_subspans: attempt to detect nested entities (threshold must be configured)
        :param threshold: the confidence threshold used to detect nested entities
        :param entity_splitter: to detect non-contiguous entities if provided
        :param strip_re: passed to :class:`~kazu.steps.ner.tokenized_word_processor.TokenizedWordProcessor`
        """
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
        self.activation_fn = cast(
            Callable[[Tensor], Tensor], sigmoid if detect_subspans else partial(softmax, dim=-1)
        )
        self.tokenized_word_processor = TokenizedWordProcessor(
            detect_subspans=detect_subspans,
            confidence_threshold=threshold,
            id2label=self.id2labels_from_label_list(labels),
            strip_re=strip_re,
        )

    @document_batch_step
    def __call__(self, docs: list[Document]) -> None:
        loader, id_section_map = self.get_dataloader(docs)
        # need this so mypy knows to expect the dataset to have encodings
        dataset = cast(HFDataset, loader.dataset)
        # run the transformer and get results
        activations = self.get_activations(loader)
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

    def get_activations(self, loader: DataLoader) -> Tensor:
        """Get a namedtuple_values_indices consisting of confidence and labels for a
        given dataloader (i.e. run bert)

        :param loader:
        :return:
        """
        with torch.no_grad():
            results = torch.cat(tuple(self.model(**batch).logits for batch in loader))
            return self.activation_fn(results)

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
        dataset = HFDataset(batch_encoding)
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

    @staticmethod
    def id2labels_from_label_list(labels: Iterable[str]) -> dict[int, str]:
        return {idx: label for idx, label in enumerate(labels)}
