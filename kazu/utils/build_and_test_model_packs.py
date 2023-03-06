import argparse
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import DictConfig

from kazu.modelling.annotation.acceptance_test import (
    execute_full_pipeline_acceptance_test,
    check_annotation_consistency,
)
from kazu.modelling.ontology_matching import assemble_pipeline
from kazu.pipeline import load_steps_and_log_memory_usage
from kazu.utils.constants import HYDRA_VERSION_BASE
from kazu.utils.utils import Singleton

logger = logging.getLogger(__name__)


@dataclass
class BuildConfiguration:
    """Dataclass that controls how a base model pack and config should be merged with a
    target model pack.
    """

    #: should this model pack use the base config as a starting point?
    requires_base_config: bool
    #: what model directories should this model pack include from the base model pack?
    models: List[str]
    #: what ontologies should this model pack use from the base ontology pack?
    #: Arg should be a list of strings to the ontology root, from the root of the base pack
    ontologies: List[str]
    #: does this model pack have its own config dir? (if used with use_base_config
    #: these will override any config files from the base config)
    has_own_config: bool
    #: should acceptance tests be run?
    run_acceptance_tests: bool = False
    #: should consistency checks be run on the gold standard?
    run_consistency_checks: bool = False
    #: Whether a base model pack is required to build this model pack
    #: This will be set automatically based on the values of the other fields,
    #: it's not available to set when instantiating the class.
    requires_base_model_pack: bool = field(init=False)

    def __post_init__(self):
        if len(self.models) > 0 or len(self.ontologies) > 0:
            self.requires_base_model_pack = True
        else:
            self.requires_base_model_pack = False


class ModelPackBuildError(Exception):
    pass


class ModelPackBuilder:
    @staticmethod
    def load_build_configuration(model_pack_path: Path) -> BuildConfiguration:
        """
        try to load a merge configuration from the model pack root. The merge configuration should be
        a json file called base_model_merge_config.json. None is returned if no model pack is found

        :param model_pack_path:
        :return:
        """
        config_path = model_pack_path.joinpath("build_config.json")
        if not config_path.exists():
            raise ModelPackBuildError(f"no merge config found at {str(config_path)}")

        with open(config_path, "r") as f:
            data = json.load(f)
        return BuildConfiguration(**data)

    @staticmethod
    def apply_merge_configurations(
        build_config: BuildConfiguration,
        maybe_base_model_pack_path: Optional[Path],
        maybe_base_configuration_path: Optional[Path],
        model_pack_build_path: Path,
        uncached_model_pack_path: Path,
    ):

        # copy the target pack to the target build dir
        shutil.copytree(
            str(uncached_model_pack_path), str(model_pack_build_path), dirs_exist_ok=True
        )
        # copy base config if required
        if build_config.requires_base_config:
            if maybe_base_configuration_path is None:
                raise ModelPackBuildError(
                    f"merge config asked for base configuration path but none was provided: {build_config}"
                )
            model_pack_build_path.joinpath("conf").mkdir(exist_ok=True)
            shutil.copytree(
                str(maybe_base_configuration_path),
                str(model_pack_build_path.joinpath("conf")),
                dirs_exist_ok=True,
            )
        if build_config.has_own_config:
            # and copy target config to override required elements (if required)
            shutil.copytree(
                str(uncached_model_pack_path.joinpath("conf")),
                str(model_pack_build_path.joinpath("conf")),
                dirs_exist_ok=True,
            )

        if maybe_base_model_pack_path is None and build_config.requires_base_model_pack:
            raise ModelPackBuildError(
                f"merge config asked for base model pack path but none was provided: {build_config}"
            )
        elif maybe_base_model_pack_path is not None and build_config.requires_base_model_pack:
            ModelPackBuilder.copy_base_model_pack_resources_to_target(
                build_config, maybe_base_model_pack_path, model_pack_build_path
            )

    @staticmethod
    def copy_base_model_pack_resources_to_target(
        build_config: BuildConfiguration,
        base_model_pack_path: Path,
        model_pack_build_path: Path,
    ):
        for model in build_config.models:
            model_source_path = base_model_pack_path.joinpath(model)
            target_dir = model_pack_build_path.joinpath(model_source_path.name)
            shutil.copytree(str(model_source_path), str(target_dir), dirs_exist_ok=True)
        for ontology_path_str in build_config.ontologies:
            ontology_path = base_model_pack_path.joinpath(ontology_path_str)
            target_path = model_pack_build_path.joinpath(ontology_path_str)
            if ontology_path.is_dir():
                shutil.copytree(str(ontology_path), str(target_path), dirs_exist_ok=True)
            else:
                shutil.copy(ontology_path, target_path)

    @staticmethod
    def clear_cached_resources_from_model_pack_dir(model_pack_path: Path):
        """
        delete any cached data from the input path

        :param model_pack_path:
        :return:
        """
        for root, d_names, f_names in os.walk(model_pack_path):
            for d_name in d_names:
                if (
                    d_name.startswith("cached")
                    or d_name.startswith("tfidf")
                    or d_name.startswith("spacy_pipeline")
                ):
                    deletion_path = os.path.join(root, d_name)
                    logger.info(f"deleting cached resource: {deletion_path}")
                    shutil.rmtree(deletion_path)

    @staticmethod
    def zip_model_pack(model_pack_name: str, build_dir: Path):
        """
        call the zip subprocess to compress model pack (requires zip on CLI)
        also moves it to parent dir

        :param model_pack_name:
        :param build_dir:
        :return:
        """
        logger.info(f"zipping model pack {model_pack_name}")
        parent_directory = build_dir.parent
        model_pack_name_with_version = model_pack_name.removesuffix(".zip")
        subprocess.run(
            # make a symlink so the top-level directory in the resulting zip file
            # has the version of the model pack in it
            ["ln", "-s", build_dir.name, model_pack_name_with_version],
            cwd=parent_directory,
        )
        subprocess.run(
            ["zip", "-r", model_pack_name, model_pack_name_with_version], cwd=parent_directory
        )

    @staticmethod
    def build_all_model_packs(
        maybe_base_model_pack_path: Optional[Path],
        maybe_base_configuration_path: Optional[Path],
        model_pack_paths: List[Path],
        zip_pack: bool,
        output_dir: Path,
        skip_tests: bool,
    ):
        """
        build multiple model packs

        :param maybe_base_model_pack_path: Path to the base model pack, if required
        :param maybe_base_configuration_path: Path to the base configuration, if required
        :param model_pack_paths: list of paths to model pack resources
        :param zip_pack: should the pack be zipped at the end?
        :param output_dir: directory to build model packs in
        :param skip_tests: don't run any tests
        :return:
        """
        if not output_dir.is_dir():
            raise ModelPackBuildError(f"{str(output_dir)} is not a directory")
        if len(os.listdir(output_dir)) > 0:
            raise ModelPackBuildError(f"{str(output_dir)} is not empty")

        kazu_version = (
            subprocess.check_output("pip show kazu | grep Version", shell=True)
            .decode("utf-8")
            .split(" ")[1]
            .strip()
        )

        for model_pack_path in model_pack_paths:
            ModelPackBuilder.reset_singletons()
            logger.info(f"building model pack at {model_pack_path}")
            ModelPackBuilder.process_model_pack_path(
                maybe_base_model_pack_path=maybe_base_model_pack_path,
                maybe_base_configuration_path=maybe_base_configuration_path,
                kazu_version=kazu_version,
                zip_pack=zip_pack,
                uncached_model_pack_path=model_pack_path,
                build_dir=output_dir,
                skip_tests=skip_tests,
            )

    @staticmethod
    def reset_singletons():
        """
        this is required between different instantiations of the pipeline config, as singleton states
        may conflict

        :return:
        """
        logger.info("clearing singletons")
        Singleton.clear_all()

    @staticmethod
    def process_model_pack_path(
        maybe_base_model_pack_path: Optional[Path],
        maybe_base_configuration_path: Optional[Path],
        kazu_version: str,
        zip_pack: bool,
        uncached_model_pack_path: Path,
        build_dir: Path,
        skip_tests: bool,
    ) -> Path:
        """
        run all configured options on a given model pack path

        :param maybe_base_model_pack_path: if this pack requires the base model pack, specify path
        :param maybe_base_configuration_path: if this pack requires the base configuration, specify path
        :param kazu_version: version of kazu used to generate model pack
        :param zip_pack: should model pack be zipped?
        :param uncached_model_pack_path: path to model pack to process
        :param build_dir: directory pack should be built in
        :param skip_tests: don't run any tests
        :return:
        """

        model_pack_build_path = build_dir.joinpath(uncached_model_pack_path.name)
        model_pack_build_path.mkdir()

        build_config = ModelPackBuilder.load_build_configuration(uncached_model_pack_path)

        ModelPackBuilder.apply_merge_configurations(
            build_config=build_config,
            maybe_base_model_pack_path=maybe_base_model_pack_path,
            maybe_base_configuration_path=maybe_base_configuration_path,
            model_pack_build_path=model_pack_build_path,
            uncached_model_pack_path=uncached_model_pack_path,
        )

        ModelPackBuilder.clear_cached_resources_from_model_pack_dir(model_pack_build_path)
        # set the env param so that hydra conf is correctly configured
        os.environ["KAZU_MODEL_PACK"] = str(model_pack_build_path)
        with initialize_config_dir(
            version_base=HYDRA_VERSION_BASE, config_dir=str(model_pack_build_path.joinpath("conf"))
        ):
            cfg = compose(
                config_name="config",
                overrides=[],
            )
            ModelPackBuilder.build_caches(cfg)
            if not skip_tests:
                if build_config.run_consistency_checks:
                    check_annotation_consistency(cfg)
                if build_config.run_acceptance_tests:
                    execute_full_pipeline_acceptance_test(cfg)
            if zip_pack:
                model_pack_name = f"{uncached_model_pack_path.name}-v{kazu_version}.zip"
                ModelPackBuilder.zip_model_pack(model_pack_name, model_pack_build_path)
        return model_pack_build_path

    @staticmethod
    def build_caches(cfg: DictConfig) -> None:
        """
        execute all processed required to build model pack caches

        :param cfg:
        :return:
        """
        parsers = instantiate(cfg.ontology_parser).values()
        explosion_path = Path(os.environ["KAZU_MODEL_PACK"]).joinpath("spacy_pipeline")
        assemble_pipeline.main(
            parsers=parsers,
            use_curations=True,
            output_dir=explosion_path,
        )
        load_steps_and_log_memory_usage(cfg)


if __name__ == "__main__":
    description = """Build and test model packs. This script takes a list of model
pack directories, and creates model packs with a number of options. Depending on
how it is called, one or more of the following may be required:

1) a conf dir (Hydra configuration for the model pack), containing
 a) configurations for a pipeline to be created
 b) a LabelStudioManager configuration to run acceptance tests or consistency checks, pointing to a running
  and network-accessible instance of LabelStudio
2) an acceptance_criteria.json in the path <model_pack_root>/acceptance_criteria.json in order for the
  acceptance tests to run. This file should specify the thresholds per NER class/Ontology for NER/linking
  respectively (see the provided model pack for an example)
3) a merge config in the path <model_pack_root>/base_model_merge_config.json, which determines which elements
  of the base configuration and model pack should be used
"""

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--base_model_pack_path",
        type=Path,
        required=False,
        help="path to base model pack, if required",
    )
    parser.add_argument(
        "--base_configuration_path",
        type=Path,
        required=False,
        help="""Path to the base configuration to use if a model pack requires it (generally
        this should be <kazu repo root>/kazu/conf)""",
    )
    parser.add_argument(
        "--model_packs_to_build",
        type=Path,
        nargs="+",
        required=True,
        help="paths to model packs to build",
    )
    parser.add_argument(
        "--zip_model_pack",
        action="store_true",
        help="should the model pack be zipped? (requires command line tool 'zip')",
    )
    parser.add_argument(
        "--model_pack_output_path",
        type=Path,
        required=True,
        help="create model packs at this location",
    )
    parser.add_argument(
        "--skip_tests",
        action="store_true",
        help="don't run any tests",
    )
    parser.add_argument(
        "--logging_config_path",
        type=Path,
        required=False,
        help="path to a logging config file, if required",
    )

    args = parser.parse_args()
    if args.logging_config_path:
        logging.config.fileConfig(args.logging_config_path)

    ModelPackBuilder.build_all_model_packs(
        maybe_base_model_pack_path=args.base_model_pack_path,
        maybe_base_configuration_path=args.base_configuration_path,
        model_pack_paths=args.model_packs_to_build,
        zip_pack=args.zip_model_pack,
        output_dir=args.model_pack_output_path,
        skip_tests=args.skip_tests,
    )
