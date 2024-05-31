"""Written for use in the release github workflow after we had issues with memory usage.

Functionally equivalent to kazu.utils.build_and_test_model_packs.ModelPackBuilder.run_acceptance_tests.
"""

import json
import logging
from os import environ
from pathlib import Path

from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

from kazu.annotation.acceptance_test import (
    analyse_full_pipeline,
    analyse_annotation_consistency,
    acceptance_criteria,
)
from kazu.annotation.label_studio import LSToKazuConversion
from kazu.utils.constants import HYDRA_VERSION_BASE


if __name__ == "__main__":
    model_pack_path = Path(environ["KAZU_MODEL_PACK"])

    # our annotations expect URI stripping, so if cleanup actions
    # are specified, ensure this is configured.
    # Note, this may cause unexpected behaviour if CleanupActions
    # is configured with anything other than 'default'?
    with initialize_config_dir(
        version_base=HYDRA_VERSION_BASE,
        config_dir=str(model_pack_path.joinpath("conf")),
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "hydra/job_logging=none",
                "hydra/hydra_logging=none",
                "CleanupActions=[default,uri_stripping]",
            ],
        )

    with model_pack_path.joinpath("build_config.json").open(mode="r") as build_config_f:
        build_config = json.load(build_config_f)

    acceptance_test_json_path = build_config["acceptance_test_json_path"]

    with model_pack_path.joinpath(acceptance_test_json_path).open(mode="r") as f:
        docs_json = json.load(f)
        docs = LSToKazuConversion.convert_tasks_to_docs(docs_json)
        analyse_annotation_consistency(docs)
        logging.info("instantiating pipeline for acceptance tests")
        # need to reinstantiate pipeline with modified config.
        pipeline = instantiate(cfg.Pipeline, _convert_="all")
        analyse_full_pipeline(pipeline, docs, acceptance_criteria=acceptance_criteria())
