import argparse
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from logging.config import fileConfig
from pathlib import Path
from typing import List, Optional
import ray
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import DictConfig


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
    def __init__(
        self,
        logging_config_path: Optional[Path],
        target_model_pack_path: Path,
        kazu_version: str,
        build_dir: Path,
        maybe_base_model_pack_path: Optional[Path],
        maybe_base_configuration_path: Optional[Path],
        skip_tests: bool,
        zip_pack: bool,
    ):
        """A ModelPackBuilder is a helper class to assist in the building of a model pack.

        .. danger::
           WARNING! since this class will configure the kazu global cache, executing
           multiple builds within the same python process could potentially lead to the
           pollution of the cache. This is because the KAZU_MODEL_PACK env variable
           is modified by this object, which should normally not happen. Rather
           than instantiating this object directly, one should instead use
           :func:`.build_all_model_packs`\\, which will control this process for you.


        :param logging_config_path: passed to :func:`logging.config.fileConfig`
        :param target_model_pack_path: path to model pack to process
        :param kazu_version: version of kazu used to generate model pack
        :param build_dir: build the pack in this directory
        :param maybe_base_model_pack_path: if this pack requires the base model pack, specify path
        :param maybe_base_configuration_path: if this pack requires the base configuration, specify path
        :param skip_tests: don't run any tests
        :param zip_pack: zip the pack at the end (requires the 'zip' CLI tool)
        """
        if logging_config_path is not None:
            fileConfig(logging_config_path)
        self.logger = logging.getLogger(__name__)
        self.zip_pack = zip_pack
        self.skip_tests = skip_tests
        self.maybe_base_configuration_path = maybe_base_configuration_path
        self.maybe_base_model_pack_path = maybe_base_model_pack_path
        self.build_dir = build_dir
        self.kazu_version = kazu_version
        self.target_model_pack_path = target_model_pack_path
        self.model_pack_build_path = self.build_dir.joinpath(self.target_model_pack_path.name)
        os.environ["KAZU_MODEL_PACK"] = str(self.model_pack_build_path)
        self.build_config = self.load_build_configuration()

    def __repr__(self):
        """For nice log messages."""
        return f"ModelPackBuilder({self.target_model_pack_path.name})"

    def build_model_pack(self) -> Path:
        """Execute the build process.

        :return: path of new pack
        """

        self.logger.info("building model pack at %s", self.model_pack_build_path)
        self.model_pack_build_path.mkdir()
        self.apply_merge_configurations()

        self.clear_cached_resources_from_model_pack_dir()
        # local import so the cache is correctly configured with KAZU_MODEL_PACK
        from kazu.utils.constants import HYDRA_VERSION_BASE

        with initialize_config_dir(
            version_base=HYDRA_VERSION_BASE,
            config_dir=str(self.model_pack_build_path.joinpath("conf")),
        ):
            cfg = compose(
                config_name="config",
                overrides=["hydra/job_logging=none", "hydra/hydra_logging=none"],
            )
            ModelPackBuilder.build_caches(cfg)
            if not self.skip_tests:
                # local import so the cache is correctly configured with KAZU_MODEL_PACK
                from kazu.modelling.annotation.acceptance_test import (
                    execute_full_pipeline_acceptance_test,
                    check_annotation_consistency,
                )

                if self.build_config.run_consistency_checks:
                    check_annotation_consistency(cfg)
                if self.build_config.run_acceptance_tests:
                    execute_full_pipeline_acceptance_test(cfg)
            if self.zip_pack:
                self.zip_model_pack()

        return self.model_pack_build_path

    def load_build_configuration(self) -> BuildConfiguration:
        """
        try to load a merge configuration from the model pack root. The merge configuration should be
        a json file called base_model_merge_config.json. None is returned if no model pack is found

        :param model_pack_path:
        :return:
        """
        config_path = self.target_model_pack_path.joinpath("build_config.json")
        if not config_path.exists():
            raise ModelPackBuildError(f"no merge config found at {str(config_path)}")

        with open(config_path, "r") as f:
            data = json.load(f)
        return BuildConfiguration(**data)

    def apply_merge_configurations(self):

        # copy the target pack to the target build dir
        shutil.copytree(
            str(self.target_model_pack_path), str(self.model_pack_build_path), dirs_exist_ok=True
        )
        # copy base config if required
        if self.build_config.requires_base_config:
            if self.maybe_base_configuration_path is None:
                raise ModelPackBuildError(
                    f"merge config asked for base configuration path but none was provided: {self.build_config}"
                )
            self.model_pack_build_path.joinpath("conf").mkdir(exist_ok=True)
            shutil.copytree(
                str(self.maybe_base_configuration_path),
                str(self.model_pack_build_path.joinpath("conf")),
                dirs_exist_ok=True,
            )
        if self.build_config.has_own_config:
            # and copy target config to override required elements (if required)
            shutil.copytree(
                str(self.target_model_pack_path.joinpath("conf")),
                str(self.model_pack_build_path.joinpath("conf")),
                dirs_exist_ok=True,
            )

        if self.maybe_base_model_pack_path is None and self.build_config.requires_base_model_pack:
            raise ModelPackBuildError(
                f"merge config asked for base model pack path but none was provided: {self.build_config}"
            )
        elif (
            self.maybe_base_model_pack_path is not None
            and self.build_config.requires_base_model_pack
        ):
            self.copy_base_model_pack_resources_to_target()

    def copy_base_model_pack_resources_to_target(self):
        assert isinstance(self.maybe_base_model_pack_path, Path)

        for model in self.build_config.models:

            model_source_path = self.maybe_base_model_pack_path.joinpath(model)
            target_dir = self.model_pack_build_path.joinpath(model_source_path.name)
            shutil.copytree(str(model_source_path), str(target_dir), dirs_exist_ok=True)
        for ontology_path_str in self.build_config.ontologies:
            ontology_path = self.maybe_base_model_pack_path.joinpath(ontology_path_str)
            target_path = self.model_pack_build_path.joinpath(ontology_path_str)
            if ontology_path.is_dir():
                shutil.copytree(str(ontology_path), str(target_path), dirs_exist_ok=True)
            else:
                shutil.copy(ontology_path, target_path)

    def clear_cached_resources_from_model_pack_dir(self):
        """Delete any cached data from the input path.

        :param model_pack_path:
        :return:
        """
        # local import so the cache is correctly configured with KAZU_MODEL_PACK
        from kazu.utils.caching import kazu_disk_cache

        kazu_disk_cache.clear()

        maybe_spacy_pipeline = self.model_pack_build_path.joinpath("spacy_pipeline")
        if maybe_spacy_pipeline.exists():
            shutil.rmtree(maybe_spacy_pipeline)

    def zip_model_pack(self):
        """
        call the zip subprocess to compress model pack (requires zip on CLI)
        also moves it to parent dir

        :param model_pack_name:
        :param build_dir:
        :return:
        """
        model_pack_name = f"{self.model_pack_build_path.name}-v{self.kazu_version}.zip"
        self.logger.info(f"zipping model pack {model_pack_name}")
        parent_directory = self.model_pack_build_path.parent
        model_pack_name_with_version = model_pack_name.removesuffix(".zip")
        subprocess.run(
            # make a symlink so the top-level directory in the resulting zip file
            # has the version of the model pack in it
            ["ln", "-s", self.model_pack_build_path.name, model_pack_name_with_version],
            cwd=parent_directory,
        )
        subprocess.run(
            ["zip", "-r", model_pack_name, model_pack_name_with_version], cwd=parent_directory
        )

    @staticmethod
    def build_caches(cfg: DictConfig) -> None:
        """
        execute all processed required to build model pack caches

        :param cfg:
        :return:
        """
        parsers = instantiate(cfg.ontology_parser).values()
        explosion_path = Path(os.environ["KAZU_MODEL_PACK"]).joinpath("spacy_pipeline")
        # local import so the cache is correctly configured with KAZU_MODEL_PACK
        from kazu.modelling.ontology_matching import assemble_pipeline

        assemble_pipeline.main(
            parsers=parsers,
            use_curations=True,
            output_dir=explosion_path,
        )
        from kazu.pipeline import load_steps_and_log_memory_usage

        load_steps_and_log_memory_usage(cfg)


@ray.remote(num_cpus=1)
class ModelPackBuilderActor(ModelPackBuilder):
    pass


def build_all_model_packs(
    maybe_base_model_pack_path: Optional[Path],
    maybe_base_configuration_path: Optional[Path],
    model_pack_paths: List[Path],
    zip_pack: bool,
    output_dir: Path,
    skip_tests: bool,
    logging_config_path: Optional[Path],
    max_parallel_build: Optional[int],
):
    """Build multiple model packs.

    :param maybe_base_model_pack_path: Path to the base model pack, if required
    :param maybe_base_configuration_path: Path to the base configuration, if required
    :param model_pack_paths: list of paths to model pack resources
    :param zip_pack: should the packs be zipped at the end?
    :param output_dir: directory to build model packs in
    :param skip_tests: don't run any tests
    :param logging_config_path: passed to logging.config.fileConfig
    :param max_parallel_build: build at most this many model packs simultaneously.
        If None, use all available CPUs
    :return:
    """
    if not output_dir.is_dir():
        raise ModelPackBuildError(f"{str(output_dir)} is not a directory")
    if len(list(output_dir.iterdir())) > 0:
        raise ModelPackBuildError(f"{str(output_dir)} is not empty")

    kazu_version = (
        subprocess.check_output("pip show kazu | grep Version", shell=True)
        .decode("utf-8")
        .split(" ")[1]
        .strip()
    )
    runtime_env = {"env_vars": {"PL_DISABLE_FORK": str(1), "TOKENIZERS_PARALLELISM": "false"}}
    ray.init(num_cpus=max_parallel_build, runtime_env=runtime_env)
    futures = []

    for model_pack_path in model_pack_paths:
        builder = ModelPackBuilderActor.remote(  # type: ignore[attr-defined]
            logging_config_path=logging_config_path,
            maybe_base_model_pack_path=maybe_base_model_pack_path,
            maybe_base_configuration_path=maybe_base_configuration_path,
            kazu_version=kazu_version,
            zip_pack=zip_pack,
            target_model_pack_path=model_pack_path,
            build_dir=output_dir,
            skip_tests=skip_tests,
        )
        futures.append(builder.build_model_pack.remote())

    unfinished = futures
    while unfinished:
        # Returns the first ObjectRef that is ready.
        finished, unfinished = ray.wait(unfinished, num_returns=1, timeout=45.0 * 60.0)
        result = ray.get(finished[0])
        print(f"model pack {result} build complete")


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
    parser.add_argument(
        "--max_parallel_build",
        type=int,
        required=False,
        help="build at most this many model packs simultaneously. If None, use all available CPUs",
    )

    args = parser.parse_args()

    build_all_model_packs(
        maybe_base_model_pack_path=args.base_model_pack_path,
        maybe_base_configuration_path=args.base_configuration_path,
        model_pack_paths=args.model_packs_to_build,
        zip_pack=args.zip_model_pack,
        output_dir=args.model_pack_output_path,
        skip_tests=args.skip_tests,
        logging_config_path=args.logging_config_path,
        max_parallel_build=args.max_parallel_build,
    )
