"""Use this script to test the model with custom text inputs and visualize the
predictions in Label Studio."""

import json
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
from kazu.training.train_multilabel_ner import _select_keys_to_use
from kazu.training.train_script import create_wrapper
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

    with open(Path(prediction_config.path) / "config.json", "r") as file:
        model_config = json.load(file)
        id2label = {int(idx): label for idx, label in model_config["id2label"].items()}
        label_list = [label for _, label in sorted(id2label.items())]
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
        manager.update(documents, 0, has_gs=False)


if __name__ == "__main__":
    main()
