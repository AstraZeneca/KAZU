from typing import Any
from collections.abc import Iterator

from torch.utils.data import IterableDataset
from transformers import BatchEncoding


class HFDataset(IterableDataset[dict[str, Any]]):
    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "input_ids": self.encodings.data["input_ids"][index],
            "attention_mask": self.encodings.data["attention_mask"][index],
            "token_type_ids": self.encodings.data["token_type_ids"][index],
        }

    def __init__(self, encodings: BatchEncoding):
        """Simple implementation of :class:`torch.utils.data.IterableDataset`\\
        , producing HF tokenizer input_id.

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
