"""Use this script to test the trained model on a held out test set.

Also visualise with LabelStudio
"""

import json
import time
from pathlib import Path

import hydra
import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig

from kazu.data import Document
from kazu.pipeline import Pipeline
from kazu.steps.ner.hf_token_classification import (
    TransformersModelForTokenClassificationNerStep,
)
from kazu.steps.ner.tokenized_word_processor import TokenizedWordProcessor
from kazu.training.config import PredictionConfig
from kazu.training.modelling_utils import (
    chunks,
    create_wrapper,
    doc_yielder,
    get_label_list_from_model,
)
from kazu.training.train_multilabel_ner import (
    _select_keys_to_use,
    calculate_metrics,
    move_entities_to_metadata,
)
from kazu.utils.constants import HYDRA_VERSION_BASE


def save_out_predictions(output_dir: Path, documents: list[Document]) -> None:
    for doc in documents:
        file_path = output_dir / f"{doc.idx}.json"
        with file_path.open("w") as f:
            f.write(doc.to_json())


@hydra.main(
    version_base=HYDRA_VERSION_BASE,
    config_path=str(
        Path(__file__).parent.parent.parent / "scripts/examples/conf/multilabel_ner_evaluate"
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
    docs_in_batch = 10
    for documents_batch in tqdm.tqdm(
        chunks(documents, docs_in_batch), total=len(documents) // docs_in_batch
    ):
        pipeline(documents_batch)
    print(f"Predicted {len(documents)} documents in {time.time() - start:.2f} seconds.")

    Path(cfg.predictions_dir).mkdir(parents=True, exist_ok=True)
    save_out_predictions(Path(cfg.predictions_dir), documents)

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
