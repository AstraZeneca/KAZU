from typing import Any, Optional

from pytorch_lightning import LightningModule
from transformers import AutoModelForTokenClassification, AutoModel


class PLAutoModelForTokenClassification(LightningModule):
    def __init__(self, model: AutoModelForTokenClassification, *args: Any, **kwargs: Any):
        """
        very simple Lightning wrapper for AutoModelForTokenClassification

        :param model: instance of AutoModelForTokenClassification
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.model = model

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self.model(**batch.data)


class PLAutoModel(LightningModule):
    def __init__(self, model: AutoModel, *args: Any, **kwargs: Any):
        """
        very simple Lightning wrapper for AutoModel

        :param model: instance of AutoModel
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.model = model

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        result = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        return result
