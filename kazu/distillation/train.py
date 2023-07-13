from typing import Union

import hydra
import pytorch_lightning
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from kazu.distillation.models import (
    SequenceTaggingDistillationForFinalLayer,
    SequenceTaggingDistillationForIntermediateLayer,
)
from kazu.utils.constants import HYDRA_VERSION_BASE


@hydra.main(version_base=HYDRA_VERSION_BASE, config_path="../../conf", config_name="config")
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
