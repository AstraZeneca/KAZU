"""Use this script to test the model with custom text inputs and visualize the
predictions in Label Studio."""

from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from kazu.data import Document
from kazu.pipeline import Pipeline
from kazu.steps.ner.hf_token_classification import (
    TransformersModelForTokenClassificationNerStep,
)
from kazu.steps.ner.tokenized_word_processor import TokenizedWordProcessor
from kazu.training.config import PredictionConfig
from kazu.training.modelling_utils import create_wrapper, get_label_list_from_model
from kazu.training.train_multilabel_ner import _select_keys_to_use
from kazu.utils.constants import HYDRA_VERSION_BASE


@hydra.main(
    version_base=HYDRA_VERSION_BASE,
    config_path=str(
        Path(__file__).parent.parent.parent / "scripts/examples/conf/multilabel_ner_predict"
    ),
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    prediction_config: PredictionConfig = instantiate(cfg.prediction_config)

    label_list = get_label_list_from_model(Path(prediction_config.path) / "config.json")
    print(f"There are {len(label_list)} labels.")

    step = TransformersModelForTokenClassificationNerStep(
        path=str(Path(prediction_config.path).absolute()),
        batch_size=prediction_config.batch_size,
        stride=prediction_config.stride,
        max_sequence_length=prediction_config.max_sequence_length,
        tokenized_word_processor=TokenizedWordProcessor(labels=label_list, use_multilabel=True),
        keys_to_use=_select_keys_to_use(prediction_config.architecture),
        device=prediction_config.device,
    )
    pipeline = Pipeline(steps=[step])
    documents = [Document.create_simple_document(text) for text in cfg.texts]
    pipeline(documents)

    manager = create_wrapper(cfg, label_list)
    if manager is not None:
        manager.update(documents, "custom_predictions", has_gs=False)


if __name__ == "__main__":
    main()
