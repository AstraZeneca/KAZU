import logging
from typing import Union

import hydra
import pytorch_lightning
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from kazu.modelling.distillation.models import (
    SequenceTaggingDistillationForFinalLayer,
    SequenceTaggingDistillationForIntermediateLayer,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config")
def start(cfg: DictConfig) -> None:

    if torch.backends.cudnn.is_available():  # for reproducibility
        torch.backends.cudnn.deterministic = cfg.DistillationTraining.cudnn.deterministic
        torch.backends.cudnn.benchmark = (
            cfg.DistillationTraining.cudnn.benchmark  # set true if the model is not for research; True will make training faster
        )

    pytorch_lightning.seed_everything(cfg.DistillationTraining.seed)
    trainer: Trainer = instantiate(cfg.DistillationTraining.trainer)
    model: Union[
        SequenceTaggingDistillationForFinalLayer, SequenceTaggingDistillationForIntermediateLayer
    ] = instantiate(cfg.DistillationTraining.model)
    trainer.fit(model)


if __name__ == "__main__":
    start()
