import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, NamedTuple

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.utilities.types import (
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS,
    EPOCH_OUTPUT,
)
from pytorch_metric_learning import miners, losses
from torch import optim
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

from kazu.utils.link_index import (
    EmbeddingIndex,
    MatMulTensorEmbeddingIndex,
)

logger = logging.getLogger(__name__)


@dataclass
class SapbertDataCollatorWithPadding:
    """
    data collator to be used with HFSapbertPairwiseDataset
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List) -> Tuple[BatchEncoding, BatchEncoding]:
        query_toks1 = [x["query_toks1"] for x in features]
        query_toks1 = self.tokenizer.pad(
            query_toks1,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        query_toks2 = [x["query_toks2"] for x in features]
        query_toks2 = self.tokenizer.pad(
            query_toks2,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        return query_toks1, query_toks2


def init_hf_collate_fn(tokenizer: Callable) -> Callable:
    """
    get a standard HF DataCollatorWithPadding, with padding=PaddingStrategy.LONGEST
    :param tokenizer:
    :return:
    """
    collate_func = DataCollatorWithPadding(tokenizer=tokenizer, padding=PaddingStrategy.LONGEST)
    return collate_func


class HFSapbertInferenceDataset(Dataset):
    """
    A dataset to be used for inferencing. In addition to standard BERT  encodings, this
    uses an 'indices' encoding that can be used to track the vector index of an embedding.
    This is needed in a multi GPU environment
    """

    def __getitem__(self, index) -> Dict[str, Any]:
        query_toks1 = {
            "input_ids": self.encodings.data["input_ids"][index],
            "token_type_ids": self.encodings.data["token_type_ids"][index],
            "attention_mask": self.encodings.data["attention_mask"][index],
            "indices": self.encodings.data["indices"][index],
        }
        return query_toks1

    def __init__(self, encodings: BatchEncoding):
        """
        simple implementation of IterableDataset, producing HF tokenizer input_id
        :param encodings:
        """
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.encodings)


class HFSapbertPairwiseDataset(Dataset):
    """
    Dataset used for training SapBert.
    """

    def __getitem__(self, index) -> Dict[str, Any]:
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
        return len(self.encodings_1.encodings)


class SapbertTrainingParams(BaseModel):
    topk: int  # score this many nearest neighbours against target
    lr: float  # learning rate
    weight_decay: float
    miner_margin: float  # passed to TripletMarginMiner
    type_of_triplets: str  # passed to TripletMarginMiner
    train_file: str  # a parquet file with three columns - syn1, syn2 and id
    train_batch_size: int
    num_workers: int  # passed to dataloaders


def get_embedding_dataloader_from_strings(
    texts: List[str], tokenizer: Callable, batch_size: int, num_workers: int
):
    """
    get a dataloader with dataset HFSapbertInferenceDataset and DataCollatorWithPadding. This should be used to
    generate embeddings for strings of interest
    :param texts: strings to use in the dataset
    :param tokenizer:
    :param batch_size:
    :param num_workers:
    :return:
    """
    indices = [i for i in range(len(texts))]
    # padding handled by collate func
    batch_encodings = tokenizer(texts, padding=False)
    batch_encodings["indices"] = indices
    dataset = HFSapbertInferenceDataset(batch_encodings)
    collate_func = init_hf_collate_fn(tokenizer)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_func,
        num_workers=num_workers,
    )
    return loader


class SapbertEvaluationDataset(NamedTuple):
    """
    to evaluate a given embedding model, we need a query datasource (i.e. things that need to be linked)]
    and an ontology datasource (i.e. things that we need to generate an embedding space for, that can be queried
    against)
    each should have three columns: default_label (text), iri (ontology id) and source (ontology name)
    """

    query_source: pd.DataFrame
    ontology_source: pd.DataFrame


class Candidate(NamedTuple):
    """
    a knowledgebase entry
    """

    default_label: str
    iri: str
    correct: bool


class GoldStandardExample(NamedTuple):
    gold_default_label: str
    gold_iri: str
    candidates: List[
        Candidate
    ]  # candidates (aka nearest neighbours) associate with this gold instance


class SapbertEvaluationDataManager:
    """
    manages the loading/parsing of multiple evaluation datasets. Each dataset should have two sources, a query source
    and an ontology source. these are then converted into data loaders, while maintaining a reference to the
    embedding metadata that should be used for evaluation

    self.dataset is Dict[dataset_name,SapbertEvaluationDataset] after construction
    """

    def __init__(self, sources: Dict[str, List[str]], debug=False):
        self.datasets: Dict[str, SapbertEvaluationDataset] = {}
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
    """
    pytorch lightning production implementation of SapBert

    Original source
    https://github.com/cambridgeltl/sapbert

    Credit

    @inproceedings{liu2021self,
        title={Self-Alignment Pretraining for Biomedical Entity Representations},
        author={Liu, Fangyu and Shareghi, Ehsan and Meng, Zaiqiao and Basaldella, Marco and Collier, Nigel},
        booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
        pages={4228--4238},
        month = jun,
        year={2021}
    }

    """

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

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokeniser = AutoTokenizer.from_pretrained(model_name_or_path, config=self.config)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.sapbert_evaluation_manager = sapbert_evaluation_manager
        self.sapbert_training_params = sapbert_training_params
        if sapbert_training_params is not None:
            self.loss = losses.MultiSimilarityLoss(alpha=1, beta=60, base=0.5)
            self.miner = miners.TripletMarginMiner(
                margin=sapbert_training_params.miner_margin,
                type_of_triplets=sapbert_training_params.type_of_triplets,
            )
            self.ontology_embeddings = None

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            [
                {"params": self.model.parameters()},
            ],
            lr=self.sapbert_training_params.lr,
            weight_decay=self.sapbert_training_params.weight_decay,
        )
        return optimizer

    def forward(self, batch):
        """
        for inference
        :param batch: standard bert input, with an additional 'indices' for representing the location of the embedding
        :return:
        """
        indices = batch.pop("indices")
        batch_embeddings = self.model(**batch)
        batch_embeddings = batch_embeddings.last_hidden_state[:, 0, :]  # cls token
        # put index as dict key so we can realign the embedding space
        return {
            index.item(): batch_embeddings[[batch_index], :]
            for batch_index, index in enumerate(indices)
        }

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
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
        return self.loss(query_embed, labels, hard_pairs)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        assert self.sapbert_training_params is not None
        training_df = pd.read_parquet(self.sapbert_training_params.train_file)
        labels = training_df["id"].astype("category").cat.codes.values
        encodings_1 = self.tokeniser(training_df["syn1"].tolist())
        encodings_2 = self.tokeniser(training_df["syn2"].tolist())
        encodings_1["labels"] = labels
        encodings_2["labels"] = labels
        train_set = HFSapbertPairwiseDataset(
            labels=labels, encodings_1=encodings_1, encodings_2=encodings_2
        )
        train_loader = torch.utils.data.DataLoader(
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
        dataloaders = []
        assert self.sapbert_evaluation_manager is not None
        assert self.sapbert_training_params is not None
        for query_source, ontology_source in self.sapbert_evaluation_manager.datasets.values():
            query_dataloader = get_embedding_dataloader_from_strings(
                texts=query_source["default_label"].tolist(),
                tokenizer=self.tokeniser,
                batch_size=self.sapbert_training_params.train_batch_size,
                num_workers=self.sapbert_training_params.num_workers,
            )
            ontology_dataloader = get_embedding_dataloader_from_strings(
                texts=ontology_source["default_label"].tolist(),
                tokenizer=self.tokeniser,
                batch_size=self.sapbert_training_params.train_batch_size,
                num_workers=self.sapbert_training_params.num_workers,
            )
            dataloaders.append(query_dataloader)
            dataloaders.append(ontology_dataloader)

    def validation_step(self, batch, batch_idx, dataset_idx) -> Optional[STEP_OUTPUT]:
        return self(batch)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self(batch)

    def get_embeddings(self, output: List[Dict[int, torch.Tensor]]) -> torch.Tensor:
        """
        get a tensor of embeddings in original order
        :param output: List[Dict[int, torch.Tensor]] int is the original index of the input (i.e. what comes out of
                        self.forward
        :return:
        """
        full_dict = {}
        for batch_id, batch in enumerate(output):
            full_dict.update(batch)
        if len(full_dict) > 1:
            embedding = torch.squeeze(torch.cat(list(full_dict.values())))
        else:
            embedding = torch.cat(list(full_dict.values()))
        return embedding

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """
        lightning override
        generate new embeddings for each SapbertEvaluationDataset.ontology_source, and query them with
        SapbertEvaluationDataset.query_source
        :param outputs:
        :return:
        """
        assert self.sapbert_evaluation_manager is not None
        outputs_all = self.all_gather(outputs)
        if self.trainer.is_global_zero:
            # stride 2 as each return from the validation step will have results for a query_output and
            # an ontology_output
            for dataset_index, output_index in enumerate(range(0, len(outputs_all), 2)):
                dataset_name = list(self.sapbert_evaluation_manager.datasets.keys())[dataset_index]
                query_output, ontology_output = (
                    outputs_all[output_index],
                    outputs_all[output_index + 1],
                )
                ontology_embeddings = self.get_embeddings(ontology_output)
                index = MatMulTensorEmbeddingIndex(name=dataset_name)
                ontology_source = self.sapbert_evaluation_manager.datasets[
                    dataset_name
                ].ontology_source
                index.add(embeddings=ontology_embeddings, metadata_df=ontology_source)
                query_embeddings = self.get_embeddings(query_output)
                queries = self.generate_evaluation_data(dataset_name, index, query_embeddings)
                metrics = self.evaluate_topk_acc(queries)
                self.log_results(dataset_name, metrics)

    def generate_evaluation_data(
        self, dataset_name: str, ontology_embeddings: EmbeddingIndex, query_embeddings: torch.Tensor
    ) -> List[GoldStandardExample]:
        """
        for a given dataset, return a list of GoldStandardExample that can be used for evaluation
        :param dataset_name: string of dataset
        :param ontology_embeddings: EmbeddingIndex to query
        :param query_embeddings: to query with!
        :return:
        """
        assert self.sapbert_evaluation_manager is not None
        assert self.sapbert_training_params is not None
        responses = []
        for i in range(query_embeddings.shape[0]):
            query_embedding = query_embeddings[[i], :]
            ontology_entry: pd.Series = self.sapbert_evaluation_manager.datasets[
                dataset_name
            ].query_source.iloc[i]
            candidate_df_slice = ontology_embeddings.search(
                query_embedding, top_n=self.sapbert_training_params.topk
            )
            default_label = ontology_entry["default_label"]
            golden_iri = ontology_entry["iri"]
            dict_candidates = self.get_candidate_dict(
                candidate_df_slice,
                golden_iri,
            )
            gold_example = GoldStandardExample(
                gold_default_label=default_label,
                gold_iri=golden_iri,
                candidates=dict_candidates,
            )
            responses.append(gold_example)
        return responses

    def log_results(self, dataset_name, metrics):
        for key, val in metrics.items():
            if key.startswith("acc"):
                self.log(key, value=val, rank_zero_only=True)
                logger.info(f"{dataset_name}: {key}, {val}")

    def get_candidate_dict(self, np_candidates: pd.DataFrame, golden_iri: str) -> List[Candidate]:
        """
        for a dataframe of candidates, return a List[Candidate]
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

    def evaluate_topk_acc(self, queries: List[GoldStandardExample]):
        """
        for a List[GoldStandardExample], get a dictionary of accuracy results at different levels of k
        (nearest neighbours)
        :param queries:
        :return:
        """
        k = len(queries[0].candidates)
        result = {}
        for i in range(0, k):
            hit = 0
            for query in queries:
                candidates = query.candidates[: i + 1]  # to get acc@(i+1)
                if any([candidate.correct for candidate in candidates]):
                    hit += 1
            result["acc{}".format(i + 1)] = hit / len(queries)

        return result

    def get_embeddings_for_strings(
        self, texts: List[str], trainer: Optional[Trainer] = None, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        convenience function: for a list of strings, generate embeddings
        :param texts:
        :param trainer: an optional PL Trainer to use. If not specified, uses the default one
        :param batch size: optional batch size to use. If not specified, use 16
        :return: a 2d tensor of embeddings
        """
        if trainer is None:
            trainer = Trainer()
        if batch_size is None:
            batch_size = 16

        loader = get_embedding_dataloader_from_strings(texts, self.tokeniser, batch_size, 0)
        results = self.get_embeddings_from_dataloader(loader, trainer)
        return results

    def get_embeddings_from_dataloader(self, loader: DataLoader, trainer: Trainer) -> torch.Tensor:
        """
        get the cls token output from all data in a dataloader as a 2d tensor
        :param loader:
        :param trainer: the PL Trainer to use
        :return: 2d tensor of cls  output
        """
        self.eval()
        results = trainer.predict(model=self, dataloaders=loader, return_predictions=True)
        results = self.get_embeddings(results)
        return results


@hydra.main(config_path="../../../conf", config_name="config")
def start(cfg: DictConfig) -> None:
    trainer: Trainer = instantiate(cfg.SapBertTraining.trainer)
    model: PLSapbertModel = instantiate(cfg.SapBertTraining.model)
    trainer.fit(model)


if __name__ == "__main__":
    start()
