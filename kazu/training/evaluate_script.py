"""Use this script to test the trained model on a held out test set.

Also visualise with LabelStudio
"""

import json
import time
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from kazu.pipeline import Pipeline
from kazu.steps.ner.hf_token_classification import (
    TransformersModelForTokenClassificationNerStep,
)
from kazu.steps.ner.tokenized_word_processor import TokenizedWordProcessor
from kazu.training.config import PredictionConfig
from kazu.training.train_multilabel_ner import (
    _select_keys_to_use,
    calculate_metrics,
    move_entities_to_metadata,
)
from kazu.training.train_script import create_wrapper, doc_yielder
from kazu.utils.constants import HYDRA_VERSION_BASE


@hydra.main(
    version_base=HYDRA_VERSION_BASE,
    config_path=str(
        Path(__file__).parent.parent.parent / "scripts/examples/conf/multilabel_ner_evaluate"
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
        tokenized_word_processor=TokenizedWordProcessor(
            labels=label_list, use_multilabel=prediction_config.use_multilabel
        ),
        keys_to_use=_select_keys_to_use(prediction_config.architecture),
        device=prediction_config.device,
    )
    pipeline = Pipeline(steps=[step])

    documents = list(doc_yielder(cfg.test_data))
    print(f"Loaded {len(documents)} documents.")
    documents = move_entities_to_metadata(documents)
    print("Predicting with the KAZU pipeline")
    start = time.time()
    pipeline(documents)
    print(f"Predicted {len(documents)} documents in {time.time() - start:.2f} seconds.")

    print("Calculating metrics")
    metrics, _ = calculate_metrics(0, documents, label_list)
    with open(Path(prediction_config.path) / "test_metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    print("Visualising with Label Studio if available")
    manager = create_wrapper(cfg, label_list)
    if manager is not None:
        manager.update(documents, "eval_on_test_set")


if __name__ == "__main__":
    main()
