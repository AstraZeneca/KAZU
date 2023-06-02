from typing import Any, Optional

from pytorch_lightning import LightningModule
from transformers import PreTrainedModel


class PLAutoModelForTokenClassification(LightningModule):
    def __init__(self, model: PreTrainedModel, *args: Any, **kwargs: Any):
        """Very simple Lightning wrapper for AutoModelForTokenClassification.

        :param model: A pretrained model for token classification - usually created with
            AutoModelForTokenClassification.from_pretrained
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.model = model

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        """Implementation of
        :external+pytorch_lightning:ref:`LightningModule.predict_step
        </common/lightning_module.rst#predict-step>`\\ ."""
        return self.model(**batch.data)


class PLAutoModel(LightningModule):
    def __init__(self, model: PreTrainedModel, *args: Any, **kwargs: Any):
        """Very simple Lightning wrapper for AutoModel.

        :param model: A pretrained model - usually created with AutoModel.from_pretrained
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.model = model

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        """Implementation of
        :external+pytorch_lightning:ref:`LightningModule.predict_step
        </common/lightning_module.rst#predict-step>`\\ ."""
        result = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        return result
