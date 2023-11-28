import logging
from typing import TypedDict, Protocol

import torch
from tokenizers import Encoding
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    BatchEncoding,
    DataCollatorWithPadding,
)
from transformers.file_utils import PaddingStrategy

from kazu.utils.utils import Singleton

logger = logging.getLogger(__name__)


class _BatchEncodingData(TypedDict):
    """Necessary because transformers doesn't distribute type information.

    We've only declared the properties we need currently, there may be more.
    """

    input_ids: dict[int, Tensor]
    token_type_ids: dict[int, Tensor]
    attention_mask: dict[int, Tensor]
    indices: dict[int, Tensor]


class _BatchEncodingFastTokenized(Protocol):
    """Necessary because transformers doesn't distribute type information.

    Also, we want to be able to say we know this was encoded with a fast tokenizer, so
    encodings won't be None.
    """

    @property
    def encodings(self) -> list[Encoding]:
        raise NotImplementedError

    @property
    def data(self) -> _BatchEncodingData:
        raise NotImplementedError


class HFSapbertInferenceDataset(Dataset[dict[str, Tensor]]):
    """A dataset to be used for inferencing.

    In addition to standard BERT  encodings, this uses an 'indices' encoding that can be
    used to track the vector index of an embedding. This is needed in a multi GPU
    environment.
    """

    def __init__(self, encodings: BatchEncoding):
        """Simple implementation of IterableDataset, producing HF tokenizer input_id.

        :param encodings: Expected to be produced by a :class:`transformers.PreTrainedTokenizerFast`.
            'slow' tokenizers (:class:`transformers.PreTrainedTokenizer`) store their encodings
            differently and so won't work with this class as-is.
        """
        if not encodings.is_fast:
            raise TypeError(
                "Your encodings come from a 'slow' tokenizer, only 'fast' tokenizers are supported."
            )
        self.encodings: _BatchEncodingFastTokenized = encodings

    def __len__(self) -> int:
        return len(self.encodings.encodings)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        query_toks1 = {
            "input_ids": self.encodings.data["input_ids"][index],
            "token_type_ids": self.encodings.data["token_type_ids"][index],
            "attention_mask": self.encodings.data["attention_mask"][index],
            "indices": self.encodings.data["indices"][index],
        }
        return query_toks1


class SapBertHelper(metaclass=Singleton):
    """Helper class to wrap useful SapBert inference functions.

    Original source:

    https://github.com/cambridgeltl/sapbert

    Licensed under MIT

    Copyright (c) Facebook, Inc. and its affiliates.

    .. raw:: html

        <details>
        <summary>Full License</summary>

    MIT License

    Copyright (c) Facebook, Inc. and its affiliates.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    .. raw:: html

        </details>

    Paper:

    | Fangyu Liu, Ehsan Shareghi, Zaiqiao Meng, Marco Basaldella, and Nigel Collier. 2021.
    | `Self-alignment pretraining for biomedical entity representations. <https://www.aclweb.org/anthology/2021.naacl-main.334.pdf>`_
    | In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4228–4238.

    .. raw:: html

        <details>
        <summary>Bibtex Citation Details</summary>

    .. code:: bibtex

        @inproceedings{liu2021self,
            title={Self-Alignment Pretraining for Biomedical Entity Representations},
            author={Liu, Fangyu and Shareghi, Ehsan and Meng, Zaiqiao and Basaldella, Marco and Collier, Nigel},
            booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
            pages={4228--4238},
            month = jun,
            year={2021}
        }

    .. raw:: html

        </details>
    """

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
        cls_tokens = batch_embeddings.last_hidden_state[:, 0, :]
        # put index as dict key so we can realign the embedding space
        return {
            index.item(): cls_tokens[[batch_index], :] for batch_index, index in enumerate(indices)
        }

    def get_embedding_dataloader_from_strings(
        self,
        texts: list[str],
        batch_size: int,
        num_workers: int,
        max_length: int = 50,
    ) -> DataLoader[BatchEncoding]:
        """Get a dataloader with dataset :class:`.HFSapbertInferenceDataset` and
        DataCollatorWithPadding.

        This should be used to generate embeddings for strings of interest.

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
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=DataCollatorWithPadding(
                tokenizer=self.tokenizer, padding=PaddingStrategy.LONGEST
            ),
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
