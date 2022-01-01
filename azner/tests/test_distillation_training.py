import tempfile

from hydra import initialize_config_module, compose

from azner.tests.utils import TEST_ASSETS_PATH
from azner.modelling.distillation.train import start

BERT_TEST_MODEL_PATH = TEST_ASSETS_PATH.joinpath("bert_test_model")
DATA_DIR = TEST_ASSETS_PATH.joinpath("tinybern")


def test_stage_2_tinybert_distillation():
    with tempfile.TemporaryDirectory() as f:
        with initialize_config_module(config_module="azner.conf"):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"DistillationTraining.model.student_model_path={BERT_TEST_MODEL_PATH}",
                    f"DistillationTraining.model.teacher_model_path={BERT_TEST_MODEL_PATH}",
                    f"DistillationTraining.model.data_dir={DATA_DIR}",
                    f"DistillationTraining.save_dir={f}",
                    "DistillationTraining.training_params.max_epochs=2",
                ],
            )

            start(cfg)
