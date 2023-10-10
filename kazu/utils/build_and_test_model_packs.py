import argparse
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from logging.config import fileConfig
from pathlib import Path
from typing import Optional

import ray
from hydra import initialize_config_dir, compose
from omegaconf import DictConfig

from kazu import __version__ as kazu_version


@dataclass
class BuildConfiguration:
    """Dataclass that controls how a base model pack and config should be
    merged with a target model pack."""

    #: should this model pack use the base config as a starting point?
    requires_base_config: bool
    #: what resource directories should this model pack include?
    #: structure is <parent_directory>:[paths within parent]
    resources: dict[str, list[str]]
    #: does this model pack have its own config dir? (if used with use_base_config
    #: these will override any config files from the base config)
    has_own_config: bool
    #: should acceptance tests be run?
    run_acceptance_tests: bool = False
    #: should consistency checks be run on the gold standard?
    run_consistency_checks: bool = False
    #: Whether resources (e.g. model binaries) are required to build this model pack
    #: This will be set automatically based on the values of the other fields,
    #: it's not available to set when instantiating the class.
    requires_resources: bool = field(init=False)

    def __post_init__(self):
        if len(self.resources) > 0:
            self.requires_resources = True
        else:
            self.requires_resources = False


class ModelPackBuildError(Exception):
    pass


class ModelPackBuilder:
    def __init__(
        self,
        logging_config_path: Optional[Path],
        target_model_pack_path: Path,
        kazu_version: str,
        build_dir: Path,
        maybe_base_configuration_path: Optional[Path],
        skip_tests: bool,
        zip_pack: bool,
    ):
        """A ModelPackBuilder is a helper class to assist in the building of a
        model pack.

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
        from kazu.steps.other.cleanup import StripMappingURIsAction

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
                # our annotations expect URI stripping, so if cleanup actions
                # are specified, ensure this is configured.
                # Note, this may cause unexpected behaviour if CleanupActions
                # is configured with anything other than 'default'?
                if "CleanupActions" in cfg:
                    if StripMappingURIsAction.__name__ not in cfg.CleanupActions:
                        self.logger.warning(
                            "%s will be overridden for acceptance tests.",
                            StripMappingURIsAction.__name__,
                        )
                        cfg = compose(
                            config_name="config",
                            overrides=[
                                "hydra/job_logging=none",
                                "hydra/hydra_logging=none",
                                "CleanupActions=[default,uri_stripping]",
                            ],
                        )

                # local import so the cache is correctly configured with KAZU_MODEL_PACK
                from kazu.annotation.acceptance_test import (
                    execute_full_pipeline_acceptance_test,
                    check_annotation_consistency,
                )

                if self.build_config.run_consistency_checks:
                    check_annotation_consistency(cfg)
                if self.build_config.run_acceptance_tests:
                    execute_full_pipeline_acceptance_test(cfg)
            self.report_tested_dependencies()
            if self.zip_pack:
                self.zip_model_pack()

        return self.model_pack_build_path

    def load_build_configuration(self) -> BuildConfiguration:
        """Try to load a build configuration from the model pack root.

        The merge configuration should be a json file called build_config.json.

        :raises ModelPackBuildError: if the merge config isn't found at the
            expected path
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

        if self.build_config.requires_resources:
            self.copy_resources_to_target()

    def copy_resources_to_target(self):

        for parent_dir_str, resource_list in self.build_config.resources.items():
            parent_dir_path = Path(parent_dir_str)
            assert parent_dir_path.is_dir(), f"{parent_dir_str} is not a directory"
            for resource_path in resource_list:
                full_path = parent_dir_path.joinpath(resource_path).absolute()
                target_dir = self.model_pack_build_path.joinpath(resource_path)
                if full_path.is_dir():
                    shutil.copytree(full_path, target_dir, dirs_exist_ok=True)
                else:
                    shutil.copy(full_path, target_dir)

    def clear_cached_resources_from_model_pack_dir(self):
        """Delete any cached data from the input path.

        :return:
        """
        # local import so the cache is correctly configured with KAZU_MODEL_PACK
        from kazu.utils.caching import kazu_disk_cache

        kazu_disk_cache.clear()

        maybe_spacy_pipeline = self.model_pack_build_path.joinpath("spacy_pipeline")
        if maybe_spacy_pipeline.exists():
            shutil.rmtree(maybe_spacy_pipeline)

    def zip_model_pack(self):
        """Call the zip subprocess to compress model pack (requires zip on CLI)
        also moves it to parent dir.

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
        """Execute all processed required to build model pack caches.

        :param cfg:
        :return:
        """
        from kazu.pipeline import load_steps_and_log_memory_usage

        load_steps_and_log_memory_usage(cfg)

    def report_tested_dependencies(self):
        dependencies = subprocess.check_output("pip freeze", shell=True).decode("utf-8")
        with self.model_pack_build_path.joinpath("tested_dependencies.txt").open(mode="w") as f:
            f.write(dependencies)


@ray.remote(num_cpus=1)
class ModelPackBuilderActor(ModelPackBuilder):
    pass


def build_all_model_packs(
    maybe_base_configuration_path: Optional[Path],
    model_pack_paths: list[Path],
    zip_pack: bool,
    output_dir: Path,
    skip_tests: bool,
    logging_config_path: Optional[Path],
    max_parallel_build: Optional[int],
) -> None:
    """Build multiple model packs.

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

    runtime_env = {"env_vars": {"PL_DISABLE_FORK": str(1), "TOKENIZERS_PARALLELISM": "false"}}
    ray.init(
        num_cpus=max_parallel_build,
        runtime_env=runtime_env,
        configure_logging=False,
        log_to_driver=True,
    )
    max_parallel_build = (
        max_parallel_build if max_parallel_build is not None else ray.cluster_resources()["CPU"]
    )
    futures: list[ray.ObjectRef] = []
    for model_pack_path in model_pack_paths:
        builder = ModelPackBuilderActor.remote(  # type: ignore[attr-defined]
            logging_config_path=logging_config_path,
            maybe_base_configuration_path=maybe_base_configuration_path,
            kazu_version=kazu_version,
            zip_pack=zip_pack,
            target_model_pack_path=model_pack_path,
            build_dir=output_dir,
            skip_tests=skip_tests,
        )
        futures.append(builder.build_model_pack.remote())
        while len(futures) >= max_parallel_build:
            futures = wait_for_model_pack_completion(futures)

    while len(futures) != 0:
        futures = wait_for_model_pack_completion(futures)


def wait_for_model_pack_completion(futures: list[ray.ObjectRef]) -> list[ray.ObjectRef]:
    # Returns the first ObjectRef that is ready.
    finished, futures = ray.wait(futures, num_returns=1, timeout=180.0 * 60.0)
    result = ray.get(finished[0])
    print(f"model pack {result} build complete")
    return futures


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
3) a merge config in the path <model_pack_root>/build_config.json, which determines which elements
  of the base configuration and model pack should be used

Note, if the environment variable KAZU_MODEL_PACK_BUILD_RESOURCES_PATH is set, this script will change it's
working directory to that location before doing anything. This will change the interpretation of relative
paths in a model packs build_config.json.
"""

    working_dir_str = os.getenv("KAZU_MODEL_PACK_BUILD_RESOURCES_PATH")
    if working_dir_str is not None:
        path = Path(working_dir_str)
        if path.is_dir():
            print(
                f"KAZU_MODEL_PACK_BUILD_RESOURCES_PATH is set to {working_dir_str}. Changing to"
                f" this path. This will affect relative paths in build_config.json"
            )
            os.chdir(path)
        else:
            raise ValueError(
                f"KAZU_MODEL_PACK_BUILD_RESOURCES_PATH is set to {working_dir_str}, but this is not a directory"
            )

    parser = argparse.ArgumentParser(description=description)

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
        maybe_base_configuration_path=args.base_configuration_path,
        model_pack_paths=args.model_packs_to_build,
        zip_pack=args.zip_model_pack,
        output_dir=args.model_pack_output_path,
        skip_tests=args.skip_tests,
        logging_config_path=args.logging_config_path,
        max_parallel_build=args.max_parallel_build,
    )
