import logging
import numpy as np
import torch
import argparse

from torch.nn import CrossEntropyLoss, MSELoss
import pytorch_lightning as pl

from transformers import AdamW
from transformers import (
    get_constant_schedule,
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup, 
    get_constant_schedule_with_warmup
    )
from transformers.utils import check_min_version

from .transformersTBPL import TinyBertForSequenceTagging
from .transformersTBPL import TinyBertForSequenceClassification
from .metrics import numeric_label_f1_score

check_min_version("4.0.0") # at least 4.0.0... for optimerzers

logger = logging.getLogger(__name__)

SCHEDULES = {
    None:       get_constant_schedule,
    "none":     get_constant_schedule,
    "warmup_cosine": get_cosine_schedule_with_warmup,
    "warmup_constant": get_constant_schedule_with_warmup,
    "warmup_linear": get_linear_schedule_with_warmup,
    "torchStepLR" : torch.optim.lr_scheduler.StepLR
}

class TaskSpecificDistillation(pl.LightningModule):
    def __init__(
        self,
        args: argparse.Namespace,
        num_train_optimization_steps: int,
        schedule=None,
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

        self.schedule = schedule
        self.num_train_optimization_steps = num_train_optimization_steps

        if torch.backends.cudnn.is_available(): # for reproducibility
            torch.backends.cudnn.determinstic = True
            torch.backends.cudnn.benchmark = False # set true if the model is not for research; True will make training faster
        
        self.save_hyperparameters(args)

    def get_optimizer_grouped_parameters(self, student_model):
        param_optimizer = list(student_model.named_parameters())
        size = 0
        logger.info('student_model.named_parameters :')
        for n, p in student_model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()
        logger.info('Total parameters: {}'.format(size))

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return optimizer_grouped_parameters

    def configure_optimizers(self,):
        """
        Configure optimizer and learning rate scheduler. 
        """
        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(student_model=self.student_model)
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=self.args.learning_rate)

        if self.schedule in ["torchStepLR"]: # Multi-GPU : must use torch scheduler
            scheduler = SCHEDULES[self.schedule](optimizer, step_size=self.num_train_optimization_steps) # PyTorch scheduler
        else:
            scheduler = SCHEDULES[self.schedule](optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.num_train_optimization_steps)  # transformers scheduler
        
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler":lr_scheduler_config}

    def soft_cross_entropy(self, predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).mean()

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
        student_logits, student_atts, student_reps = self.student_model(
            input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
            is_student=True)
        with torch.no_grad():
            teacher_logits, teacher_atts, teacher_reps = self.teacher_model(
                input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,)

        if not self.args.pred_distill: # Intermediate-layer (Transformer, Embedding) Distillation
            raise NotImplemented
        else: # Prediction-layer Distillation
            loss = self.soft_cross_entropy(student_logits / self.args.temperature,
                                          teacher_logits / self.args.temperature)
        
        # Logging
        self.log('trainLoss', loss, prog_bar=True)
        lr_list = self.lr_schedulers().get_lr()
        self.log('lr0', lr_list[0], prog_bar=True)
        self.log('lr1', lr_list[1], prog_bar=True)

        return loss


class SequenceTaggingTaskSpecificDistillation(TaskSpecificDistillation):
    def __init__(
        self,
        args: argparse.Namespace,
        label_list: list,
        schedule: str = None,
        num_train_optimization_steps: int = None,
        metric: str = "entity_f1",
        **config_kwargs
        ):

        super().__init__(args=args, schedule=schedule, num_train_optimization_steps=num_train_optimization_steps)

        self.args = args
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.metric = metric
        
        self.teacher_model = TinyBertForSequenceTagging.from_pretrained(args.teacher_model, num_labels=self.num_labels)
        self.student_model = TinyBertForSequenceTagging.from_pretrained(args.student_model, num_labels=self.num_labels)


    def validation_step_predLayer(self, batch, batch_idx):
        input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
        with torch.no_grad():
            logits, _, _ = self.student_model(
                input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                )
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.type(torch.LongTensor).to(self.device).view(-1))
        loss = tmp_eval_loss.mean().item()

        self.log("validation_loss_step", loss) # Add sync_dist=True to sync logging across all GPU workers
        return {"loss":loss, "logits":logits.detach().cpu().numpy(), "label_ids":label_ids.detach().cpu().numpy()} 

    def validation_step(self, batch, batch_idx):
        if not self.args.pred_distill: # Intermediate-layer (Transformer, Embedding) Distillation
            raise NotImplemented
        else: # Prediction-layer Distillation
            return self.validation_step_predLayer(batch, batch_idx)

    def validation_epoch_end(self, val_step_outputs):
        epoch_loss_mean = np.mean([vStepOut["loss"] for vStepOut in val_step_outputs])
        self.log("validation_loss_epoch", epoch_loss_mean, on_step=False, on_epoch=True, sync_dist=True)

        preds_logits = np.concatenate([vStepOut["logits"] for vStepOut in val_step_outputs], axis=0)
        preds = np.argmax(preds_logits, axis=-1)
        label_ids = np.concatenate([vStepOut["label_ids"] for vStepOut in val_step_outputs], axis=0)
        if self.metric == "entity_f1":
            result = numeric_label_f1_score(preds=preds, label_ids=label_ids, label_list=self.label_list)
            self.log("valF1", result, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True) # Micro F1
        return epoch_loss_mean

    def test_step(self, batch, batch_idx):
        raise NotImplemented
