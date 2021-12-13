from pathlib import Path
import os
import json
from typing import Any, Dict, Optional, Union

import torch

from pytorch_lightning.plugins import CheckpointIO


class CustomCheckpointIO(CheckpointIO):
    """
    A plugin for saving student model (without saving teacher model)
    """

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: Union[str, Path],
        storage_options: Optional[Any] = None,
    ) -> None:
        """
        Save distilled (student) model.

        :param checkpoint: contents to save. Including `state_dict`, `optimizer_states` and `callbacks`.
        :param path:
        :param storage_options:
        """

        dirPath = os.path.dirname(path)

        output_config_file = os.path.join(dirPath, "hyper_parameters.json")
        json.dump(obj=checkpoint["hyper_parameters"], fp=open(output_config_file, "w"), indent=2)

        if os.path.basename(path) != "last.ckpt":
            output_log_file = os.path.join(dirPath, "train_callback_log.json")
            with open(output_log_file, "a") as callbackLogFP:
                jsonable_callback_log = {
                    k: str(v)
                    for k, v in checkpoint["callbacks"][
                        list(checkpoint["callbacks"].keys())[0]
                    ].items()
                }
                json.dump(obj=jsonable_callback_log, fp=callbackLogFP)
                callbackLogFP.write("\n")

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

        self.tokenizer.save_vocabulary(os.path.dirname(path))

    def load_checkpoint(
        self, path: Union[str, Path], storage_options: Optional[Any] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def remove_checkpoint(
        self,
        path: Union[str, Path],
    ) -> None:

        os.remove(path)
