import copy
import dataclasses
import json
import logging
import pickle
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Any, Union, cast
import math

import torch
from kazu.annotation.acceptance_test import score_sections, aggregate_ner_results
from kazu.data import Section, Document, PROCESSING_EXCEPTION, ENTITY_OUTSIDE_SYMBOL, NumericMetric
from kazu.pipeline import Pipeline
from kazu.steps.ner.hf_token_classification import (
    TransformersModelForTokenClassificationNerStep,
)
from kazu.steps.ner.tokenized_word_processor import TokenizedWordProcessor
from kazu.training.modelling import (
    DistilBertForMultiLabelTokenClassification,
    BertForMultiLabelTokenClassification,
)
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerFast,
    BatchEncoding,
    PreTrainedModel,
    get_scheduler,
    SchedulerType,
)

from kazu.training.config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SavedModel:
    path: Path
    step: int
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)


class ModelSaver:
    def __init__(self, save_dir: Path, max_to_keep: int = 5, patience: int = 5):
        self.patience = patience
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True)
        self.max_to_keep = max_to_keep
        self.saves: list[SavedModel] = []
        self.best: Optional[SavedModel] = None
        self.patience_check_count = 0

    @staticmethod
    def save_model(tokenizer: PreTrainedTokenizerFast, model: PreTrainedModel, path: Path) -> None:
        tokenizer.save_pretrained(path)
        model.save_pretrained(path, safe_serialization=False)

    def save(
        self,
        model: PreTrainedModel,
        step: int,
        tokenizer: PreTrainedTokenizerFast,
        metrics: dict[str, float],
        stopping_metric: str,
        eval_docs: Optional[list[Document]] = None,
    ) -> None:
        self.patience_check_count += 1
        save_dir = self.save_dir.joinpath(f"step_{step}")
        save_dir.mkdir(exist_ok=True)
        saved_model = SavedModel(path=save_dir, step=step, metrics=metrics)
        if self.best is None:
            self.best = saved_model
        elif metrics[stopping_metric] > self.best.metrics[stopping_metric]:
            self.best = saved_model
            self.patience_check_count = 0

        self.save_model(tokenizer, model, save_dir)
        with save_dir.joinpath("metrics.json").open(mode="w") as f:
            json.dump(metrics, f)
        if eval_docs:
            self._save_eval_docs(eval_docs, save_dir)

        self.saves.append(saved_model)
        if len(self.saves) > self.max_to_keep:
            popped = self.saves.pop(0)
            if popped is not self.best:
                shutil.rmtree(popped.path, ignore_errors=True)

        if self.best.metrics[stopping_metric] == 0.0:
            logger.info("patience not activated as no best model found yet")
        elif self.patience_check_count >= self.patience:
            best_path = self.best.path.parent.joinpath(f"best_{self.best.path.name}")
            shutil.move(self.best.path, best_path)
            raise RuntimeError(f"patience exceeded, best model moved to {best_path}")

    def _save_eval_docs(self, eval_docs: list[Document], save_dir: Path) -> None:
        eval_docs_dir = save_dir.joinpath("eval_docs")
        eval_docs_dir.mkdir(exist_ok=True)
        for doc in eval_docs:
            with eval_docs_dir.joinpath(f"{doc.idx}.json").open(mode="w") as f:
                f.write(doc.to_json())


class KazuNerMultiLabelTrainingDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        docs_iter: Iterable[Document],
        model_tokenizer: PreTrainedTokenizerFast,
        labels: list[str],
        tmp_dir: Path,
        use_cache: bool = True,
        max_length: int = 128,
        stride: int = 64,
        max_docs: Optional[int] = None,
        keep_doc_reference: bool = False,
    ):
        self.max_docs = max_docs
        self.stride = stride
        self.max_length = max_length
        self.tmp_dir = tmp_dir
        self.model_tokenizer = model_tokenizer
        self.labels = labels
        self.item_count = 0
        self.docs = []

        if not use_cache:
            shutil.rmtree(self.tmp_dir)
            self.tmp_dir.mkdir()
        else:
            self.item_count = len(list(self.tmp_dir.iterdir()))

        if not use_cache or keep_doc_reference:
            for doc_i, doc in tqdm(enumerate(docs_iter), desc=f"preparing data at {self.tmp_dir}"):
                if self.max_docs and doc_i >= self.max_docs:
                    break
                if keep_doc_reference:
                    self.docs.append(doc)
                if not use_cache:
                    for section in doc.sections:
                        encoding = self.tokenize_and_align(section)
                        datapoints_this_item = len(encoding["input_ids"])
                        for i in range(datapoints_this_item):
                            result: dict[str, Tensor] = {}
                            for key in encoding.keys():
                                if key != "labels":
                                    result[key] = torch.IntTensor(encoding[key][i])
                                else:
                                    result[key] = torch.FloatTensor(encoding[key][i])
                            out_path = self.tmp_dir / f"{self.item_count}.pkl"
                            with out_path.open(mode="wb") as f:
                                pickle.dump(result, f)
                            self.item_count += 1
        self.docs = move_entities_to_metadata(self.docs)

    def get_docs_copy(self) -> list[Document]:
        return copy.deepcopy(self.docs)

    def tokenize_and_align(self, section: Section) -> dict[str, Tensor]:
        starts_ends_and_ents: defaultdict[tuple[int, int], set[str]] = defaultdict(set)
        for entity in section.entities:
            starts_ends_and_ents[(entity.start, entity.end)].add(entity.entity_class)

        encoding: BatchEncoding = self.model_tokenizer(
            section.text,
            is_split_into_words=False,
            max_length=self.max_length,
            return_overflowing_tokens=True,
            stride=self.stride,
            padding="max_length",
            truncation=True,
        )

        one_hot_batch = []
        for data in encoding.encodings:
            datapoint_batch = []
            prev_word_idx = None
            for i, word_idx in enumerate(data.words):
                one_hot: list[Union[int, float]] = []
                label_set: set[str] = set()
                if word_idx is not None:
                    word_start, word_end = data.offsets[i]
                    for (
                        ent_start,
                        ent_end,
                    ), ent_labels in starts_ends_and_ents.items():
                        # TODO: check index
                        if word_start >= ent_start and word_end <= ent_end:
                            for ent_label in ent_labels:
                                label_set.add(ent_label)
                    if not label_set:
                        label_set.add(ENTITY_OUTSIDE_SYMBOL)

                for label in self.labels:
                    if word_idx is None or word_idx == prev_word_idx:
                        one_hot.append(-100)
                    elif label in label_set:
                        one_hot.append(1.0)
                    else:
                        one_hot.append(0.0)
                datapoint_batch.append(one_hot)
                prev_word_idx = word_idx
            one_hot_batch.append(datapoint_batch)
        encoding["labels"] = one_hot_batch
        return cast(dict[str, Tensor], encoding.data)

    def __len__(self) -> int:
        return self.item_count

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        with self.tmp_dir.joinpath(f"{idx}.pkl").open(mode="rb") as f:
            item: dict[str, Tensor] = pickle.load(f)

        return item


def move_entities_to_metadata(docs: list[Document]) -> list[Document]:
    task_id = 0
    for doc in docs:
        if PROCESSING_EXCEPTION in doc.metadata:
            doc.metadata.pop(PROCESSING_EXCEPTION)
        for section in doc.sections:
            if "gold_entities" not in section.metadata:
                section.metadata["gold_entities"] = section.entities
                section.metadata["label_studio_task_id"] = task_id
                section.entities = []
                task_id += 1
    return docs


class Trainer:
    def __init__(
        self,
        training_config: TrainingConfig,
        pretrained_model_name_or_path: str,
        label_list: list[str],
        train_dataset: KazuNerMultiLabelTrainingDataset,
        eval_dataset: KazuNerMultiLabelTrainingDataset,
        working_dir: Path,
        summary_writer: Optional[SummaryWriter] = None,
    ):

        self.training_config = training_config
        self.summary_writer = summary_writer
        self.working_dir = working_dir
        model_save_dir = working_dir.joinpath("models")
        model_save_dir.mkdir(exist_ok=True)
        self.saver = ModelSaver(save_dir=model_save_dir, max_to_keep=5)
        self.eval_dir = self.working_dir.joinpath("eval_tmp")
        self.eval_dir.mkdir(exist_ok=True)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.label_list = label_list
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    def _write_to_tensorboard(
        self, global_step: int, main_tag: str, tag_scalar_dict: dict[str, NumericMetric]
    ) -> None:
        if self.summary_writer:
            self.summary_writer.add_scalars(
                main_tag, tag_scalar_dict=tag_scalar_dict, global_step=global_step
            )

    def evaluate_model(
        self, model: PreTrainedModel, global_step: int, save_model: bool = True
    ) -> None:
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=self.training_config.workers,
            pin_memory=True,
        )
        epoch_loss = self._log_eval_loss(eval_dataloader, global_step, model)

        eval_docs = self._process_docs(model)
        all_results = self._calculate_metrics(epoch_loss, eval_docs, global_step)
        if save_model:
            self.saver.save(
                model,
                global_step,
                self.train_dataset.model_tokenizer,
                metrics=all_results,
                stopping_metric="mean_f1",
                eval_docs=eval_docs,
            )

    def _calculate_metrics(
        self, epoch_loss: float, eval_docs: list[Document], global_step: int
    ) -> dict[str, Any]:
        for doc in eval_docs:
            if PROCESSING_EXCEPTION in doc.metadata:
                logger.error(doc.metadata[PROCESSING_EXCEPTION])
                break
        ner_dict = score_sections(eval_docs)
        ner_results = aggregate_ner_results(ner_dict)
        all_results: dict[str, Any] = {}
        for clazz, result in ner_results.items():
            tb_results = {}
            tb_results["precision"] = result.precision
            tb_results["recall"] = result.recall
            self._write_to_tensorboard(global_step, clazz, tb_results)

            support = result.tp + result.fn

            logger.info(f"{clazz} precision: {result.precision}")
            logger.info(f"{clazz} recall: {result.recall}")
            logger.info(f"{clazz} support: {support}")

            all_results[f"{clazz}_precision"] = result.precision
            all_results[f"{clazz}_recall"] = result.recall
            all_results[f"{clazz}_support"] = support

            false_positives: defaultdict[str, dict[str, int]] = defaultdict(dict)
            for match, count in result.fp_info:
                false_positives[clazz][match] = count
            all_results["false_positives"] = dict(false_positives)
            false_negatives: defaultdict[str, dict[str, int]] = defaultdict(dict)
            for match, count in result.fn_info:
                false_negatives[clazz][match] = count
            all_results["false_negatives"] = dict(false_negatives)
        label_set = set(self.label_list)
        label_set.remove(ENTITY_OUTSIDE_SYMBOL)
        if len(ner_results) != len(label_set):
            mean_f1 = 0.0
            found_str = "\n".join(ner_results.keys())
            need_str = "\n".join(set(label_set).difference(ner_results))
            logger.info(
                f"model not considered as not all labels represented. Found: \n {found_str}\n Need: \n {need_str}"
            )
        else:
            try:
                mean_f1 = sum(
                    [
                        (2 * (x.precision * x.recall) / (x.precision + x.recall))
                        for x in ner_results.values()
                    ]
                ) / len(ner_results)
            except ZeroDivisionError:
                logger.info("model not considered as some eval values are 0")
                mean_f1 = 0.0
        all_results["eval_loss"] = epoch_loss
        all_results["mean_f1"] = mean_f1
        return all_results

    def _process_docs(self, model: PreTrainedModel) -> list[Document]:
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir_path = Path(tempdir)
            ModelSaver.save_model(self.train_dataset.model_tokenizer, model, tempdir_path)
            step = TransformersModelForTokenClassificationNerStep(
                path=str(tempdir_path.absolute()),
                batch_size=self.training_config.batch_size,
                stride=self.train_dataset.stride,
                max_sequence_length=self.train_dataset.max_length,
                tokenized_word_processor=TokenizedWordProcessor(
                    labels=self.label_list, use_multilabel=True
                ),
                keys_to_use=["input_ids", "attention_mask"],
                device=self.training_config.device,
                multi_label=True,
            )

            eval_pipeline = Pipeline(steps=[step])
            eval_docs = self.eval_dataset.get_docs_copy()
            eval_pipeline(eval_docs)
        return eval_docs

    def _log_eval_loss(
        self,
        eval_dataloader: DataLoader[dict[str, Tensor]],
        global_step: int,
        model: PreTrainedModel,
    ) -> float:
        model.eval()
        total_eval_loss = 0.0
        loss_func = torch.nn.BCEWithLogitsLoss()
        for batch in eval_dataloader:
            with torch.no_grad():
                # must pop labels for eval
                labels = batch.pop("labels")
                batch.pop("overflow_to_sample_mapping")
                batch = {k: v.to(self.training_config.device) for k, v in batch.items()}
                outputs = model(**batch)
                ignore_index = labels.mean(-1).squeeze().int()
                flat_outputs = outputs.logits.squeeze()[ignore_index != -100]
                flat_labels = labels.squeeze()[ignore_index != -100].to(self.training_config.device)
                loss = loss_func(flat_outputs, flat_labels)
                total_eval_loss += loss.item() * labels.size(0)
        epoch_loss = total_eval_loss / len(eval_dataloader)
        logger.info(f"evaluation loss: {epoch_loss}")
        self._write_to_tensorboard(global_step, "epoch_loss", {"eval_loss": epoch_loss})
        return epoch_loss

    def train_model(self) -> None:
        if len(list(self.saver.save_dir.iterdir())) != 0:
            raise RuntimeError(f"working dir {self.saver.save_dir} not empty")

        label2id = {label: i for i, label in enumerate(self.label_list)}
        id2label = {v: k for k, v in label2id.items()}

        if self.training_config.architecture == "bert":
            model = BertForMultiLabelTokenClassification.from_pretrained(
                self.pretrained_model_name_or_path,
                num_labels=len(self.label_list),
                id2label=id2label,
                label2id=label2id,
            )
        elif self.training_config.architecture == "distilbert":
            model = DistilBertForMultiLabelTokenClassification.from_pretrained(
                self.pretrained_model_name_or_path,
                num_labels=len(self.label_list),
                id2label=id2label,
                label2id=label2id,
            )
        else:
            raise ValueError(f"unknown architecture {self.training_config.architecture}")
        logger.info(f"training samples: {len(self.train_dataset)}")
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=self.training_config.workers,
            pin_memory=True,
        )
        logger.info(f"training batches: {len(train_dataloader)}")
        optimizer = AdamW(model.parameters(), lr=self.training_config.lr)
        num_training_steps = self.training_config.num_epochs * len(train_dataloader)
        progress_bar = tqdm(range(num_training_steps))
        if self.training_config.lr_scheduler_warmup_prop:
            lr_scheduler = get_scheduler(
                name=SchedulerType.COSINE_WITH_RESTARTS,
                optimizer=optimizer,
                num_warmup_steps=math.ceil(
                    len(train_dataloader) * self.training_config.lr_scheduler_warmup_prop
                ),
                num_training_steps=num_training_steps,
            )
        else:
            lr_scheduler = None
        device = "mps"
        model.to(device)
        batch_count = 0
        global_step = 0
        for epoch in range(self.training_config.num_epochs):
            model.train()
            running_loss = 0.0
            for batch in train_dataloader:
                batch.pop("overflow_to_sample_mapping")
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                if lr_scheduler:
                    lr_scheduler.step()
                optimizer.zero_grad()
                ls = loss.item()
                running_loss += ls * batch["labels"].size(0)
                progress_bar.update(1)
                progress_bar.set_description(f"loss: {ls}")
                batch_count += 1
                global_step += 1
                if global_step % 100 == 0:
                    self._write_to_tensorboard(global_step, "loss", {"train_loss": ls})

                if global_step % self.training_config.evaluate_at_step_interval == 0:
                    self.evaluate_model(model, global_step)

            epoch_loss = running_loss / len(train_dataloader)
            self._write_to_tensorboard(global_step, "epoch_loss", {"train_loss": epoch_loss})
            progress_bar.write(f"Epoch {epoch}, Loss: {epoch_loss}")
        progress_bar.close()
