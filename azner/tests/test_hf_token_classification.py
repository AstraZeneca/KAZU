from azner.data.data import SimpleDocument
from hydra import initialize_config_module, compose
from hydra.utils import instantiate

from azner.tests.utils import (
    get_TransformersModelForTokenClassificationNerStep_model_path,
    ner_test_cases,
)


def test_TransformersModelForTokenClassificationNerStep():

    with initialize_config_module(config_module="azner.conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"TransformersModelForTokenClassificationNerStep.path={get_TransformersModelForTokenClassificationNerStep_model_path()}"
            ],
        )

        step = instantiate(cfg.TransformersModelForTokenClassificationNerStep)
        docs = [SimpleDocument(x) for x in ner_test_cases()]
        successes, failures = step(docs)
        for doc in successes:
            for section in doc.sections:
                assert len(section.entities) > 0
