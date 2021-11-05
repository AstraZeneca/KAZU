from typing import Iterator
from torch.utils.data.dataset import T_co
from torch.utils.data import IterableDataset
from transformers import BatchEncoding


class HFDataset(IterableDataset):
    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, encodings: BatchEncoding):
        """
        simple implementation of IterableDataset, producing HF tokenizer input_id
        :param encodings:
        """
        self.encodings = encodings

    def __iter__(self) -> Iterator[T_co]:
        for encoding in self.encodings.data:
            yield {"input_ids": encoding["input_ids"],"attention_mask":encoding["attention_mask"]}
