import os
from pathlib import Path
from typing import Any, Optional, Union

import torch
from omegaconf import OmegaConf
from pytorch_lightning.plugins import CheckpointIO
from transformers import AutoTokenizer


# lightning doesn't distribute type information, so to mypy
# this is subclassing 'Any'.
class StudentModelCheckpointIO(CheckpointIO):  # type: ignore[misc]
    """A plugin for saving student model (without saving teacher model)"""

    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model_name_or_path = model_name_or_path

    def save_checkpoint(
        self,
        checkpoint: dict[str, Any],
        path: Union[str, Path],
        storage_options: Optional[Any] = None,
    ) -> None:
        """Save distilled (student) model. Loading currently not implemented.

        :param checkpoint: contents to save. Including ``state_dict``, ``optimizer_states`` and ``callbacks``.
        :param path:
        :param storage_options:
        """

        dirPath = os.path.dirname(path)

        output_config_file = os.path.join(dirPath, "hyper_parameters.json")
        OmegaConf.save(
            OmegaConf.create(checkpoint["hyper_parameters"]), output_config_file, resolve=True
        )
        studentModel_state_dict = {
            key[len("student_model.") :]: value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("student_model.")
        }
        teacherModel_state_dict = {
            key: value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("teacher_model.")
        }
        assert len(checkpoint["state_dict"]) == len(studentModel_state_dict) + len(
            teacherModel_state_dict
        ), "Missing structures while saving trained model."
        torch.save(studentModel_state_dict, path)

        AutoTokenizer.from_pretrained(self.model_name_or_path).save_vocabulary(
            os.path.dirname(path)
        )

    def load_checkpoint(
        self, path: Union[str, Path], storage_options: Optional[Any] = None
    ) -> dict[str, Any]:
        """Not currently implemented.

        See :external+pytorch_lightning:meth:`CheckpointIO.load_checkpoint <lightning.pytorch.plugins.io.CheckpointIO.load_checkpoint>`
        for details of the abstract method.
        """
        raise NotImplementedError

    def remove_checkpoint(
        self,
        path: Union[str, Path],
    ) -> None:

        os.remove(path)
