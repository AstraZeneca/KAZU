import logging
from typing import List, Dict, Union

import numpy as np
import pytorch_lightning as pl
import torch
from cachetools import LRUCache
from omegaconf import ListConfig, OmegaConf
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, EPOCH_OUTPUT
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
from transformers.utils import check_min_version

from azner.modelling.distillation.data_utils import printable_text
from azner.modelling.distillation.dataprocessor import NerProcessor
from azner.modelling.distillation.metrics import numeric_label_f1_score, IGNORE_IDX
from azner.modelling.distillation.tiny_transformers import TinyBertForSequenceTagging

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
    """
    A dataset used for Ner. designed for on the fly tokenisation to speed up multi processing. Uses caching to
    prevent repeated processing
    """

    def __init__(
        self, 
        tokenizer: AutoTokenizer, 
        examples: List[InputExample], 
        label_map: Dict[str, int], 
        max_length: int,
    ):
        """

        :param tokenizer: and instance of AutoTokenizer
        :param examples: a list of InputExample, typically created from a
            :class:`azner.modelling.distillation.dataprocessor.NerProcessor`
        :param label_map: str to int mapping of labels
        :param max_length: The maximum number of tokens per instance that the model can handle. 
            Inputs longer than max_length value will be truncated.
        """
        self.label_map = label_map
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.call_count = 0
        self.cache = LRUCache(5000)

    def __getitem__(self, index) -> T_co:
        if index not in self.cache:
            self.cache[index] = self.convert_single_example(
                ex_index=index, example=self.examples[index]
            )
        self.call_count += 1
        return self.cache[index]

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

        ntokens = []
        segment_ids = []
        label_id = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_id.append(IGNORE_IDX)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if labels[i] == "X":
                label_id.append(IGNORE_IDX)
            else:
                label_id.append(self.label_map[labels[i]])
        
        # Truncation
        if len(ntokens) > self.max_length-1:
            assert (len(ntokens)==len(segment_ids)) and (len(ntokens)==len(label_id))
            ntokens = ntokens[:self.max_length-1]
            segment_ids = segment_ids[:self.max_length-1]
            label_id = label_id[:self.max_length-1]

        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_id.append(IGNORE_IDX)

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        # The mask has 1 for real tokens and 0 for padding tokens.
        input_mask = [1] * len(input_ids)
        if self.call_count < 4 and ex_index < 4:  # Examples. Executed only once per model run
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([printable_text(x) for x in tokens]))
            logger.info("ntokens: %s" % " ".join([printable_text(x) for x in ntokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_id: %s" % " ".join([str(x) for x in label_id]))

        result = {
            "labels": label_id,
            "input_ids": input_ids,
            "token_type_ids": segment_ids,
            "attention_mask": input_mask,
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
    ):
        """
        Base class for distillation on PyTorch Lightning platform.

        :param temperature:
        :param warmup_steps:
        :param learning_rate:
        :param weight_decay:
        :param batch_size:
        :param accumulate_grad_batches:
        :param max_epochs:
        :param schedule:
        """

        super().__init__()
        self.accumulate_grad_batches = accumulate_grad_batches
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        self.temperature = temperature
        self.schedule = schedule
        self.training_examples = self.get_training_examples()

        self.num_train_optimization_steps = int(
            len(self.training_examples)
            / self.batch_size
            / self.accumulate_grad_batches
            * max_epochs
        )
        self.warmup_steps = warmup_steps
        logger.info(
            "num_train_optimization_steps: {}, args.warmup_steps: {}, args.gradient_accumulation_steps: {}".format(
                self.num_train_optimization_steps, self.warmup_steps, self.accumulate_grad_batches
            )
        )

    def get_training_examples(self) -> List[InputExample]:
        """
        subclasses should implement this

        :return:
        """
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
        student_logits, student_atts, student_reps = self.student_model(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            is_student=True,
        )

        #self.teacher_model.eval()
        with torch.no_grad():
            teacher_logits, teacher_atts, teacher_reps = self.teacher_model(
                input_ids=batch["input_ids"],
                token_type_ids=batch["token_type_ids"],
                attention_mask=batch["attention_mask"],
            )

        loss = self.soft_cross_entropy(
            student_logits / self.temperature, teacher_logits / self.temperature
        )

        # Logging
        self.log("training_loss", loss, prog_bar=True, on_step=True)
        lr_list = self.lr_schedulers().get_lr()
        self.log("lr0", lr_list[0], prog_bar=True, on_step=True)
        self.log("lr1", lr_list[1], prog_bar=True, on_step=True)

        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        epoch_loss_mean = torch.mean(torch.tensor([x["loss"] for x in outputs]))
        self.log("training_loss", epoch_loss_mean, on_step=False, on_epoch=True, sync_dist=True)


class SequenceTaggingTaskSpecificDistillation(TaskSpecificDistillation):
    """
    Sequence tagging implementation
    """

    def __init__(
        self,
        temperature: float,
        warmup_steps: int,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        accumulate_grad_batches: int,
        max_epochs: int,
        max_length: int,
        data_dir: str,
        label_list: Union[List, ListConfig],
        student_model_path: str,
        teacher_model_path: str,
        num_workers: int,
        schedule: str = None,
        metric: str = "entity_f1",
    ):
        """

        :param temperature:
        :param warmup_steps:
        :param learning_rate:
        :param weight_decay:
        :param batch_size:
        :param accumulate_grad_batches:
        :param max_epochs:
        :param data_dir:
        :param label_list:
        :param student_model_path:
        :param teacher_model_path:
        :param num_workers:
        :param schedule:
        :param metric:
        """
        self.processor = NerProcessor()
        self.data_dir = data_dir
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_path)
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
        if isinstance(label_list, ListConfig):
            self.label_list = OmegaConf.to_container(label_list)
        else:
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
        self.loss = CrossEntropyLoss(ignore_index=IGNORE_IDX)
        self.save_hyperparameters()

    def validation_step_predLayer(self, batch, batch_idx):
        logits, _, _ = self.student_model(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            is_student=True,
        )

        loss = (
            self.loss(
                logits.view(-1, self.num_labels),
                batch["labels"].view(-1),
            )
            .mean()
            .item()
        )

        self.log("validation_loss", loss, sync_dist=True, on_step=True, prog_bar=True)
        return {
            "loss": loss,
            "logits": logits.detach().cpu(),
            "label_ids": batch["labels"].detach().cpu(),
            "attention_mask": batch["attention_mask"].detach().cpu(),
        }

    def validation_step(self, batch, batch_idx):
        return self.validation_step_predLayer(batch, batch_idx)

    def validation_epoch_end(self, val_step_outputs):
        epoch_loss_mean = np.mean([x["loss"] for x in val_step_outputs])
        self.log("validation_loss", epoch_loss_mean, on_step=False, on_epoch=True, sync_dist=True)

        preds, golds = [], []
        for output in val_step_outputs:

            attention_mask = output["attention_mask"]
            golds.extend(self.tensor_to_jagged_array(output["label_ids"], attention_mask))
            preds.extend(
                self.tensor_to_jagged_array(torch.argmax(output["logits"], dim=-1), attention_mask)
            )

        result = numeric_label_f1_score(preds=preds, golds=golds, label_list=self.label_list)
        self.log(
            self.metric, result, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )  # Micro F1
        return epoch_loss_mean

    def tensor_to_jagged_array(
        self, tensor: torch.Tensor, attention_mask: torch.Tensor
    ) -> List[List[int]]:
        result = []
        for arr, mask in zip(tensor.numpy(), attention_mask.numpy()):
            result.append(arr[0 : mask.sum()].tolist())
        return result

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def train_dataloader(self) -> TRAIN_DATALOADERS:

        dataset = NerDataset(
            tokenizer=self.tokenizer, examples=self.training_examples, label_map=self.label_map, max_length=self.max_length
        )
        collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer, padding=True)
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=True,
        )

    def get_training_examples(self) -> List[InputExample]:
        return self.processor.get_train_examples(self.data_dir)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        examples = self.processor.get_dev_examples(self.data_dir)
        dataset = NerDataset(
            tokenizer=self.tokenizer, examples=examples, label_map=self.label_map, max_length=self.max_length
        )
        collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer, padding=True)
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=True,
        )
