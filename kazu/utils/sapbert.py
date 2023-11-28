import logging
from typing import cast

import torch
from tokenizers import Encoding
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    BatchEncoding,
    PreTrainedTokenizerBase,
    DataCollatorWithPadding,
)
from transformers.file_utils import PaddingStrategy

from kazu.utils.utils import Singleton

logger = logging.getLogger(__name__)


def init_hf_collate_fn(tokenizer: PreTrainedTokenizerBase) -> DataCollatorWithPadding:
    """Get a standard HF DataCollatorWithPadding, with padding=PaddingStrategy.LONGEST.

    :param tokenizer:
    :return:
    """
    collate_func = DataCollatorWithPadding(tokenizer=tokenizer, padding=PaddingStrategy.LONGEST)
    return collate_func


class HFSapbertInferenceDataset(Dataset[dict[str, Tensor]]):
    """A dataset to be used for inferencing.

    In addition to standard BERT  encodings, this uses an 'indices' encoding that can be
    used to track the vector index of an embedding. This is needed in a multi GPU
    environment.
    """

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        query_toks1 = {
            "input_ids": self.encodings.data["input_ids"][index],
            "token_type_ids": self.encodings.data["token_type_ids"][index],
            "attention_mask": self.encodings.data["attention_mask"][index],
            "indices": self.encodings.data["indices"][index],
        }
        return query_toks1

    def __init__(self, encodings: BatchEncoding):
        """Simple implementation of IterableDataset, producing HF tokenizer input_id.

        :param encodings:
        """
        self.encodings = encodings

    def __len__(self) -> int:
        encodings = cast(list[Encoding], self.encodings.encodings)
        return len(encodings)


class SapBertHelper(metaclass=Singleton):
    """Helper class to wrap useful SapBert inference functions."""

    def __init__(self, path: str):
        """

        :param path: passed to :class:`transformers.AutoConfig`\\, :class:`transformers.AutoTokenizer`\\, :class:`transformers.AutoModel` .from_pretrained.
        """
        self.config = AutoConfig.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, config=self.config)
        self.model = AutoModel.from_pretrained(path, config=self.config)

    @staticmethod
    def get_embeddings(output: list[dict[int, torch.Tensor]]) -> torch.Tensor:
        """Get a tensor of embeddings in original order.

        :param output: int is the original index of the input.
        :return:
        """
        full_dict = {}
        for batch in output:
            full_dict.update(batch)
        if len(full_dict) > 1:
            embedding = torch.squeeze(torch.cat(list(full_dict.values())))
        else:
            embedding = torch.cat(list(full_dict.values()))
        return embedding

    def get_embeddings_from_dataloader(self, loader: DataLoader[BatchEncoding]) -> torch.Tensor:
        """Get the cls token output from all data in a dataloader as a 2d tensor.

        :param loader:
        :return: 2d tensor of cls  output
        """
        model = self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in loader:
                predictions.append(self.get_prediction_from_batch(model, batch))
        results = self.get_embeddings(predictions)
        return results

    @staticmethod
    def get_prediction_from_batch(
        model: torch.nn.Module, batch: dict[str, Tensor]
    ) -> dict[int, Tensor]:
        """Pass the batch through the model, and return the Tensor of the cls token."""

        indices = batch.pop("indices")
        batch_embeddings = model(**batch)
        batch_embeddings = batch_embeddings.last_hidden_state[:, 0, :]  # cls token
        # put index as dict key so we can realign the embedding space
        return {
            index.item(): batch_embeddings[[batch_index], :]
            for batch_index, index in enumerate(indices)
        }

    def get_embedding_dataloader_from_strings(
        self,
        texts: list[str],
        batch_size: int,
        num_workers: int,
        max_length: int = 50,
    ) -> DataLoader[BatchEncoding]:
        """Get a dataloader with dataset :class:`.HFSapbertInferenceDataset` and
        DataCollatorWithPadding. This should be used to generate embeddings for strings
        of interest.

        :param texts: strings to use in the dataset
        :param batch_size:
        :param num_workers:
        :param max_length:
        :return:
        """
        indices = [i for i in range(len(texts))]
        # padding handled by collate func
        batch_encodings = self.tokenizer(
            texts, padding=PaddingStrategy.MAX_LENGTH, max_length=max_length, truncation=True
        )
        batch_encodings["indices"] = indices
        dataset = HFSapbertInferenceDataset(batch_encodings)
        collate_func = init_hf_collate_fn(self.tokenizer)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_func,
            num_workers=num_workers,
        )
        return loader

    def get_embeddings_for_strings(self, texts: list[str], batch_size: int = 16) -> torch.Tensor:
        """For a list of strings, generate embeddings.

        This is a convenience function for users, as we need to carry out these steps
        several times in the codebase.

        :param texts:
        :param batch_size: optional batch size to use. If not specified, use 16
        :return: a 2d tensor of embeddings
        """

        loader = self.get_embedding_dataloader_from_strings(texts, batch_size, 0)
        results = self.get_embeddings_from_dataloader(loader)
        return results
