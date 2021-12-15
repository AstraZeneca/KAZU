import argparse
import logging
from typing import List, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from transformers import AdamW, AutoTokenizer, InputExample, DataCollatorForTokenClassification
from transformers import (
    get_constant_schedule,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from transformers.file_utils import PaddingStrategy
from transformers.utils import check_min_version

from azner.modelling.distillation.distillation.transformersTBPL import TinyBertForSequenceTagging
from .metrics import numeric_label_f1_score
from ..dataprocessor.data_utils import printable_text
from ..dataprocessor.seqtag_tasks import NerProcessor

check_min_version("4.0.0")  # at least 4.0.0... for optimerzers

logger = logging.getLogger(__name__)

SCHEDULES = {
    None: get_constant_schedule,
    "none": get_constant_schedule,
    "warmup_cosine": get_cosine_schedule_with_warmup,
    "warmup_constant": get_constant_schedule_with_warmup,
    "warmup_linear": get_linear_schedule_with_warmup,
    "torchStepLR": torch.optim.lr_scheduler.StepLR,
}


class NerDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, examples: List[InputExample], label_map: Dict):
        self.label_map = label_map
        self.examples = examples
        self.tokenizer = tokenizer

    def __getitem__(self, index) -> T_co:
        return self.convert_single_example(ex_index=index, example=self.examples[index])

    def __len__(self):
        return len(self.examples)

    def convert_single_example(self, ex_index, example) -> Dict[str, torch.Tensor]:
        textlist = example.text_a.split()
        labellist = example.label.split()
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m, tok in enumerate(token):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")

        # drop if token is longer than max_seq_length
        ntokens = []
        segment_ids = []
        label_id = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_id.append(self.label_map["O"])  # putting O instead of [CLS]
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if labels[i] == "X":
                label_id.append(self.label_map["O"])
            else:
                label_id.append(self.label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_id.append(self.label_map["O"])  # putting O instead of [SEP]

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        # The mask has 1 for real tokens and 0 for padding tokens.
        input_mask = [1] * len(input_ids)
        if ex_index < 4:  # Examples before model run
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_id: %s" % " ".join([str(x) for x in label_id]))

        result = {
            "labels": torch.tensor(label_id, dtype=torch.int),
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "segment_ids": torch.tensor(segment_ids, dtype=torch.int64),
            "input_mask": torch.tensor(input_mask, dtype=torch.int64),
        }
        return result


class TaskSpecificDistillation(pl.LightningModule):
    def __init__(
        self,
        temperature: float,
        warmup_steps: int,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        accumulate_grad_batches: int,
        max_epochs: int,
        schedule=None,
        warmup_proportion: float = None,
    ):
        """
        Base class for distillation on PyTorch Lightning platform.

        :param args:
        :type args: argparse.Namespace
        :param num_train_optimization_steps: Total training steps (the number of updates of parameters).
        :type num_train_optimization_steps: int
        :param schedule: Method for learning rate schedule. One of none, "warmup_cosine", "warmup_constant", "warmup_linear". defaults to None
        :type schedule: [type], optional
        """

        super().__init__()
        self.accumulate_grad_batches = accumulate_grad_batches
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        self.temperature = temperature
        self.schedule = schedule
        self.training_examples = self.get_training_examples()

        self.num_train_optimization_steps = (
            len(self.training_examples)
            / self.batch_size
            / self.accumulate_grad_batches
            * max_epochs
        )
        self.warmup_steps = (
            int(warmup_proportion * self.num_train_optimization_steps)
            if warmup_steps is None
            else warmup_steps
        )
        logger.info(
            "num_train_optimization_steps: {}, args.warmup_steps: {}, args.gradient_accumulation_steps: {}".format(
                self.num_train_optimization_steps, self.warmup_steps, self.accumulate_grad_batches
            )
        )

    def get_training_examples(self) -> List[InputExample]:
        raise NotImplementedError()

    def get_optimizer_grouped_parameters(self, student_model):
        param_optimizer = list(student_model.named_parameters())
        size = 0
        logger.info("student_model.named_parameters :")
        for n, p in student_model.named_parameters():
            logger.info("n: {}".format(n))
            size += p.nelement()
        logger.info("Total parameters: {}".format(size))

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def configure_optimizers(
        self,
    ):
        """
        Configure optimizer and learning rate scheduler.
        """
        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(
            student_model=self.student_model
        )
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=self.learning_rate)

        if self.schedule in ["torchStepLR"]:  # Multi-GPU : must use torch scheduler
            scheduler = SCHEDULES[self.schedule](
                optimizer, step_size=self.num_train_optimization_steps
            )  # PyTorch scheduler
        else:
            scheduler = SCHEDULES[self.schedule](
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.num_train_optimization_steps,
            )  # transformers scheduler

        lr_scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def soft_cross_entropy(self, predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (-targets_prob * student_likelihood).mean()

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
        student_logits, student_atts, student_reps = self.student_model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            is_student=True,
        )
        with torch.no_grad():
            teacher_logits, teacher_atts, teacher_reps = self.teacher_model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
            )

        if not self.pred_distill:  # Intermediate-layer (Transformer, Embedding) Distillation
            raise NotImplementedError
        else:  # Prediction-layer Distillation
            loss = self.soft_cross_entropy(
                student_logits / self.temperature, teacher_logits / self.temperature
            )

        # Logging
        self.log("trainLoss", loss, prog_bar=True)
        lr_list = self.lr_schedulers().get_lr()
        self.log("lr0", lr_list[0], prog_bar=True)
        self.log("lr1", lr_list[1], prog_bar=True)

        return loss


class SequenceTaggingTaskSpecificDistillation(TaskSpecificDistillation):
    def __init__(
        self,
        temperature: float,
        warmup_steps: int,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        accumulate_grad_batches: int,
        max_epochs: int,
        data_dir: str,
        label_list: list,
        student_model_path: str,
        teacher_model_path: str,
        num_workers: int,
        schedule: str = None,
        metric: str = "entity_f1",
        **config_kwargs,
    ):

        super().__init__(
            schedule=schedule,
            accumulate_grad_batches=accumulate_grad_batches,
            temperature=temperature,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
        )

        self.num_workers = num_workers
        self.data_dir = data_dir
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.metric = metric

        self.teacher_model = TinyBertForSequenceTagging.from_pretrained(
            teacher_model_path, num_labels=self.num_labels
        )
        self.student_model = TinyBertForSequenceTagging.from_pretrained(
            student_model_path, num_labels=self.num_labels
        )
        self.processor = NerProcessor()
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_path)

    def validation_step_predLayer(self, batch, batch_idx):
        input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
        logits, _, _ = self.student_model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
        )
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(
            logits.view(-1, self.num_labels),
            label_ids.type(torch.LongTensor).to(self.device).view(-1),
        )
        loss = tmp_eval_loss.mean().item()

        self.log(
            "validation_loss_step", loss
        )  # Add sync_dist=True to sync logging across all GPU workers
        return {
            "loss": loss,
            "logits": logits.detach().cpu().numpy(),
            "label_ids": label_ids.detach().cpu().numpy(),
        }

    def validation_step(self, batch, batch_idx):
        return self.validation_step_predLayer(batch, batch_idx)

    def validation_epoch_end(self, val_step_outputs):
        epoch_loss_mean = np.mean([vStepOut["loss"] for vStepOut in val_step_outputs])
        self.log(
            "validation_loss_epoch", epoch_loss_mean, on_step=False, on_epoch=True, sync_dist=True
        )

        preds_logits = np.concatenate([vStepOut["logits"] for vStepOut in val_step_outputs], axis=0)
        preds = np.argmax(preds_logits, axis=-1)
        label_ids = np.concatenate([vStepOut["label_ids"] for vStepOut in val_step_outputs], axis=0)
        if self.metric == "entity_f1":
            result = numeric_label_f1_score(
                preds=preds, label_ids=label_ids, label_list=self.label_list
            )
            self.log(
                "valF1", result, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
            )  # Micro F1
        return epoch_loss_mean

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def train_dataloader(self) -> TRAIN_DATALOADERS:

        dataset = NerDataset(
            tokenizer=self.tokenizer, examples=self.training_examples, label_map=self.label_map
        )
        collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.student_model.config.max_length,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collator,
            pin_memory=True,
        )

    def get_training_examples(self) -> List[InputExample]:
        return self.processor.get_train_examples(self.data_dir)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        examples = self.processor.get_dev_examples(self.data_dir)
        dataset = NerDataset(tokenizer=self.tokenizer, examples=examples, label_map=self.label_map)
        collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.student_model.config.max_length,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collator,
            pin_memory=True,
        )
