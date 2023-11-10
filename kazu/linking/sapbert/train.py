import logging
from dataclasses import dataclass
from typing import cast, Any, Optional, Union, NamedTuple

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel
from tokenizers import Encoding
from torch import optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
)
from transformers.file_utils import PaddingStrategy

try:
    from pytorch_metric_learning import miners, losses
    from pytorch_lightning import Trainer, LightningModule
    from pytorch_lightning.utilities.types import (
        STEP_OUTPUT,
        TRAIN_DATALOADERS,
        EVAL_DATALOADERS,
        EPOCH_OUTPUT,
    )
except ImportError as e:
    raise ImportError(
        "Running the SapBERT model training code requires several additional dependencies to be installed.\n"
        "We recommend running 'pip install kazu[model-training]' to get all model training"
        " dependencies."
    ) from e

from kazu.utils.constants import HYDRA_VERSION_BASE
from kazu.utils.sapbert import SapBertHelper

logger = logging.getLogger(__name__)


@dataclass
class SapbertDataCollatorWithPadding:
    """Data collator to be used with HFSapbertPairwiseDataset."""

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(
        self, features: list[dict[str, BatchEncoding]]
    ) -> tuple[BatchEncoding, BatchEncoding]:
        query_toks1 = [x["query_toks1"] for x in features]
        query_toks1_enc = self.tokenizer.pad(
            query_toks1,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        query_toks2 = [x["query_toks2"] for x in features]
        query_toks2_enc = self.tokenizer.pad(
            query_toks2,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        return query_toks1_enc, query_toks2_enc


class HFSapbertPairwiseDataset(Dataset):
    """Dataset used for training SapBert."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        query_toks1 = {
            "input_ids": self.encodings_1.data["input_ids"][index],
            "token_type_ids": self.encodings_1.data["token_type_ids"][index],
            "labels": self.encodings_1.data["labels"][index],
            "attention_mask": self.encodings_1.data["attention_mask"][index],
        }
        query_toks2 = {
            "input_ids": self.encodings_2.data["input_ids"][index],
            "token_type_ids": self.encodings_2.data["token_type_ids"][index],
            "labels": self.encodings_2.data["labels"][index],
            "attention_mask": self.encodings_2.data["attention_mask"][index],
        }

        return {"query_toks1": query_toks1, "query_toks2": query_toks2}

    def __init__(self, encodings_1: BatchEncoding, encodings_2: BatchEncoding, labels: np.ndarray):
        """
        :param encodings_1: encodings for example 1
        :param encodings_2: encodings for example 2
        :param labels: labels i.e. knowledgebase identifier for both encodings, as an int
        """
        encodings_1["labels"] = labels
        encodings_2["labels"] = labels
        self.encodings_1 = encodings_1
        self.encodings_2 = encodings_2

    def __len__(self):
        encodings = cast(list[Encoding], self.encodings_1.encodings)
        return len(encodings)


class SapbertTrainingParams(BaseModel):
    topk: int  # score this many nearest neighbours against target
    lr: float  # learning rate
    weight_decay: float
    miner_margin: float  # passed to TripletMarginMiner
    type_of_triplets: str  # passed to TripletMarginMiner
    train_file: str  # a parquet file with three columns - syn1, syn2 and id
    train_batch_size: int
    num_workers: int  # passed to dataloaders


class SapbertEvaluationDataset(NamedTuple):
    """To evaluate a given embedding model, we need a query datasource (i.e. things that
    need to be linked)] and an ontology datasource (i.e. things that we need to generate
    an embedding space for, that can be queried against) each should have three columns:

    default_label (text), iri (ontology id) and source (ontology name)
    """

    query_source: pd.DataFrame
    ontology_source: pd.DataFrame


class Candidate(NamedTuple):
    """A knowledgebase entry."""

    default_label: str
    iri: str
    correct: bool


class GoldStandardExample(NamedTuple):
    gold_default_label: str
    gold_iri: str
    candidates: list[
        Candidate
    ]  # candidates (aka nearest neighbours) associate with this gold instance


class SapbertEvaluationDataManager:
    """Manages the loading/parsing of multiple evaluation datasets. Each dataset should
    have two sources, a query source and an ontology source. these are then converted
    into data loaders, while maintaining a reference to the embedding metadata that
    should be used for evaluation.

    self.dataset is dict[dataset_name, SapbertEvaluationDataset] after construction
    """

    def __init__(self, sources: dict[str, list[str]], debug: bool = False):
        self.datasets: dict[str, SapbertEvaluationDataset] = {}
        for source_name, (
            query_source_path,
            ontology_source_path,
        ) in sources.items():
            query_source = pd.read_parquet(query_source_path)
            ontology_source = pd.read_parquet(ontology_source_path)
            if debug:
                query_source = query_source.head(10)
                ontology_source = ontology_source.sample(frac=1.0).head(100)
            self.datasets[source_name] = SapbertEvaluationDataset(
                query_source=query_source, ontology_source=ontology_source
            )


class PLSapbertModel(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        sapbert_training_params: Optional[SapbertTrainingParams] = None,
        sapbert_evaluation_manager: Optional[SapbertEvaluationDataManager] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        :param model_name_or_path: passed to AutoModel.from_pretrained
        :param sapbert_training_params: optional SapbertTrainingParams, only needed if training a model
        :param sapbert_evaluation_manager: optional SapbertEvaluationDataManager, only needed if training a model
        :param args: passed to LightningModule
        :param kwargs: passed to LightningModule
        """

        super().__init__(*args, **kwargs)
        self.sapbert_helper = SapBertHelper(model_name_or_path)
        self.model = self.sapbert_helper.model
        self.sapbert_evaluation_manager = sapbert_evaluation_manager
        self.sapbert_training_params = sapbert_training_params
        if sapbert_training_params is not None:
            self.loss = losses.MultiSimilarityLoss(alpha=1, beta=60, base=0.5)
            self.miner = miners.TripletMarginMiner(
                margin=sapbert_training_params.miner_margin,
                type_of_triplets=sapbert_training_params.type_of_triplets,
            )

    def configure_optimizers(self):
        """Implementation of
        :external+pytorch_lightning:ref:`LightningModule.configure_optimizers
        </common/lightning_module.rst#configure-optimizers>`\\ ."""
        assert self.sapbert_training_params is not None
        optimizer = optim.AdamW(
            [
                {"params": self.model.parameters()},
            ],
            lr=self.sapbert_training_params.lr,
            weight_decay=self.sapbert_training_params.weight_decay,
        )
        return optimizer

    def forward(self, batch: BatchEncoding) -> dict[int, torch.Tensor]:
        """For inference.

        :param batch: standard bert input, with an additional 'indices' for representing
            the location of the embedding
        :return:
        """
        return self.sapbert_helper.get_prediction_from_batch(self.model, batch)

    def training_step(self, batch: Any, batch_idx: int, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """Implementation of
        :external+pytorch_lightning:ref:`LightningModule.training_step
        </common/lightning_module.rst#training-step>`\\ ."""
        query_toks1, query_toks2 = batch
        # labels should be identical, so we only need one
        labels = query_toks1.pop("labels")
        query_toks2.pop("labels")
        last_hidden_state1 = self.model(**query_toks1, return_dict=True).last_hidden_state
        last_hidden_state2 = self.model(**query_toks2, return_dict=True).last_hidden_state
        query_embed1 = last_hidden_state1[:, 0]  # query : [batch_size, hidden]
        query_embed2 = last_hidden_state2[:, 0]  # query : [batch_size, hidden]
        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        hard_pairs = self.miner(query_embed, labels)
        return self.loss(query_embed, labels, hard_pairs)  # type: ignore [no-any-return] # no clear type info for the loss

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Implementation of
        :external+pytorch_lightning:ref:`LightningModule.train_dataloader
        </common/lightning_module.rst#train-dataloader>`\\ ."""
        assert self.sapbert_training_params is not None
        training_df = pd.read_parquet(self.sapbert_training_params.train_file)
        labels = training_df["id"].astype("category").cat.codes.to_numpy()
        encodings_1 = self.tokeniser(training_df["syn1"].tolist())
        encodings_2 = self.tokeniser(training_df["syn2"].tolist())
        encodings_1["labels"] = labels
        encodings_2["labels"] = labels
        train_set = HFSapbertPairwiseDataset(
            labels=labels, encodings_1=encodings_1, encodings_2=encodings_2
        )
        train_loader = DataLoader(
            train_set,
            batch_size=self.sapbert_training_params.train_batch_size,
            shuffle=True,
            num_workers=self.sapbert_training_params.num_workers,
            collate_fn=SapbertDataCollatorWithPadding(
                self.tokeniser, padding=PaddingStrategy.LONGEST
            ),
        )

        return train_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Implementation of
        :external+pytorch_lightning:ref:`LightningModule.val_dataloader
        </common/lightning_module.rst#val-dataloader>`\\ ."""
        dataloaders = []
        assert self.sapbert_evaluation_manager is not None
        assert self.sapbert_training_params is not None
        for query_source, ontology_source in self.sapbert_evaluation_manager.datasets.values():
            query_dataloader = self.sapbert_helper.get_embedding_dataloader_from_strings(
                texts=query_source["default_label"].tolist(),
                batch_size=self.sapbert_training_params.train_batch_size,
                num_workers=self.sapbert_training_params.num_workers,
            )
            ontology_dataloader = self.sapbert_helper.get_embedding_dataloader_from_strings(
                texts=ontology_source["default_label"].tolist(),
                batch_size=self.sapbert_training_params.train_batch_size,
                num_workers=self.sapbert_training_params.num_workers,
            )
            dataloaders.append(query_dataloader)
            dataloaders.append(ontology_dataloader)

        return dataloaders

    def validation_step(
        self, batch: Any, batch_idx: int, dataset_idx: int
    ) -> Optional[STEP_OUTPUT]:
        """Implementation of
        :external+pytorch_lightning:ref:`LightningModule.validation_step
        </common/lightning_module.rst#validation-step>`\\ ."""
        return self(batch)  # type: ignore [no-any-return] # no type info from Pytorch Lightning

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        """Implementation of
        :external+pytorch_lightning:ref:`LightningModule.predict_step
        </common/lightning_module.rst#predict-step>`\\ ."""
        return self(batch)

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, list[EPOCH_OUTPUT]]) -> None:
        """Lightning override generate new embeddings for each
        :attr:`SapbertEvaluationDataset.ontology_source` and query them with
        :attr:`SapbertEvaluationDataset.query_source`

        :param outputs:
        :return:
        """
        assert self.sapbert_evaluation_manager is not None
        raise NotImplementedError

    def log_results(self, dataset_name, metrics):
        for key, val in metrics.items():
            if key.startswith("acc"):
                self.log(key, value=val, rank_zero_only=True)
                logger.info(f"{dataset_name}: {key}, {val}")

    def get_candidate_dict(self, np_candidates: pd.DataFrame, golden_iri: str) -> list[Candidate]:
        """Convert rows in a dataframe representing candidate KB entries into a
        corresponding :class:`Candidate` per row.

        :param np_candidates:
        :param golden_iri:
        :return:
        """
        candidates_filtered = []
        for i, np_candidate_row in np_candidates.iterrows():
            candidates_filtered.append(
                Candidate(
                    default_label=np_candidate_row["default_label"],
                    iri=np_candidate_row["iri"],
                    correct=np_candidate_row["iri"] == golden_iri,
                )
            )
        return candidates_filtered

    def evaluate_topk_acc(self, queries: list[GoldStandardExample]) -> dict[str, float]:
        """Get a dictionary of accuracy results at different levels of k (nearest
        neighbours)

        :param queries:
        :return:
        """
        k = len(queries[0].candidates)
        result = {}
        for i in range(0, k):
            hit = 0
            for query in queries:
                candidates = query.candidates[: i + 1]  # to get acc@(i+1)
                if any(candidate.correct for candidate in candidates):
                    hit += 1
            result["acc{}".format(i + 1)] = hit / len(queries)

        return result


@hydra.main(version_base=HYDRA_VERSION_BASE, config_path="../../../conf", config_name="config")
def start(cfg: DictConfig) -> None:
    trainer: Trainer = instantiate(cfg.SapBertTraining.trainer)
    model: PLSapbertModel = instantiate(cfg.SapBertTraining.model)
    trainer.fit(model)


if __name__ == "__main__":
    start()
