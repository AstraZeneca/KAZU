import argparse
import ast
import os
import shutil
import subprocess
from distutils.dir_util import copy_tree
from pathlib import Path
from typing import Tuple, List, Optional

from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from kazu.modelling.annotation.acceptance_test import (
    execute_full_pipeline_acceptance_test,
    check_annotation_consistency,
)
from kazu.modelling.ontology_matching import assemble_pipeline
from kazu.pipeline import load_steps
from kazu.utils.utils import Singleton
from omegaconf import DictConfig


class ModelPackBuildError(Exception):
    pass


class ModelPackBuilder:
    @staticmethod
    def clear_cached_resources_from_model_pack_dir(model_path_path: Path):
        """
        delete any cached data from the input path

        :param model_path_path:
        :return:
        """
        for root, d_names, f_names in os.walk(model_path_path):
            for d_name in d_names:
                if (
                    d_name.startswith("cached")
                    or d_name.startswith("tfidf")
                    or d_name.startswith("spacy_pipeline")
                ):
                    deletion_path = os.path.join(root, d_name)
                    print(f"deleting cached resource: {deletion_path}")
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
        print(f"zipping model pack {model_pack_name}")
        subprocess.run(["zip", "-r", model_pack_name, "."], cwd=build_dir)
        shutil.move(build_dir.joinpath(model_pack_name), build_dir.parent.joinpath(model_pack_name))

    @staticmethod
    def build_all_model_packs(
        maybe_base_model_pack_path: Optional[Path],
        maybe_base_configuration_path: Optional[Path],
        custom_model_pack_params: Optional[List[Tuple[Path, bool, bool]]],
        zip_pack: bool,
        run_acceptance_tests: bool,
        run_consistency_checks: bool,
        output_dir: Path,
    ):
        """
        build multiple model packs

        :param maybe_base_model_pack_path: Path to the base model pack, if required
        :param maybe_base_configuration_path: Path to the base configuration, if required
        :param custom_model_pack_params: optional tuples of (<path to custom pack resources>,
            <use base config>, <use base model pack>,)
        :param zip_pack: should the pack be zipped at the end?
        :param run_acceptance_tests: should acceptance tests be run?
        :param run_consistency_checks: run against the annotated corpus, to highlight potential annotation errors
        :param output_dir: directory to build model packs in
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

        # build base resources first
        if maybe_base_model_pack_path is not None:
            print(f"building base model pack at {maybe_base_model_pack_path}")
            base_model_pack_path_with_cached_files = ModelPackBuilder.process_model_pack_path(
                maybe_base_configuration_path=maybe_base_configuration_path,
                maybe_base_model_pack_path=None,
                kazu_version=kazu_version,
                run_consistency_checks=run_consistency_checks,
                run_acceptance_tests=run_acceptance_tests,
                zip_pack=zip_pack,
                uncached_model_pack_path=maybe_base_model_pack_path,
                build_dir=output_dir,
            )
        else:
            base_model_pack_path_with_cached_files = None

        if custom_model_pack_params is not None:
            for (
                custom_model_pack_path,
                include_default_config,
                include_default_resources,
            ) in custom_model_pack_params:
                ModelPackBuilder.reset_singletons()
                print(f"building custom model pack at {custom_model_pack_path}")
                ModelPackBuilder.process_model_pack_path(
                    maybe_base_configuration_path=maybe_base_configuration_path,
                    maybe_base_model_pack_path=base_model_pack_path_with_cached_files,
                    kazu_version=kazu_version,
                    run_consistency_checks=run_consistency_checks,
                    run_acceptance_tests=run_acceptance_tests,
                    zip_pack=zip_pack,
                    uncached_model_pack_path=custom_model_pack_path,
                    build_dir=output_dir,
                )

    @staticmethod
    def reset_singletons():
        """
        this is required between different instantiations of the pipeline config, as singleton states
        may conflict

        :return:
        """
        print("clearing singletons")
        Singleton._instances = {}

    @staticmethod
    def process_model_pack_path(
        maybe_base_configuration_path: Optional[Path],
        maybe_base_model_pack_path: Optional[Path],
        kazu_version: str,
        run_acceptance_tests: bool,
        run_consistency_checks: bool,
        zip_pack: bool,
        uncached_model_pack_path: Path,
        build_dir: Path,
    ) -> Path:
        """
        run all configured options on a given model pack path

        :param maybe_base_configuration_path: if this pack requires the base configuration, specify path
        :param maybe_base_model_pack_path: if this pack requires the base model pack, specify path
        :param kazu_version: version of kazu used to generate model pack
        :param run_acceptance_tests: should acceptance tests be run?
        :param run_acceptance_tests: should consistency_checks be run on the annotated corpus?
        :param zip_pack: should model pack be zipped?
        :param uncached_model_pack_path: path to model pack to process
        :param build_dir: directory pack should be built in
        :return:
        """

        model_pack_build_path = build_dir.joinpath(uncached_model_pack_path.name)
        model_pack_build_path.mkdir()

        if maybe_base_model_pack_path is not None:
            print(f"copying base model pack for {uncached_model_pack_path.name}")
            copy_tree(str(maybe_base_model_pack_path), str(model_pack_build_path))

        conf_path = model_pack_build_path.joinpath("conf")
        if maybe_base_configuration_path is not None:
            print(f"copying base configuration for {uncached_model_pack_path.name}")
            conf_path.mkdir(exist_ok=True)
            copy_tree(str(maybe_base_configuration_path), str(conf_path))
        else:
            # the base conf may have been copied over if the base model pack was used,
            # so we get rid of it here if required
            print(f"removing base configuration for {uncached_model_pack_path.name}")
            shutil.rmtree(conf_path)

        print(f"preparing to build model pack {model_pack_build_path.name}")
        copy_tree(str(uncached_model_pack_path), str(model_pack_build_path))
        ModelPackBuilder.clear_cached_resources_from_model_pack_dir(model_pack_build_path)
        # set the env param so that hydra conf is correctly configured
        os.environ["KAZU_MODEL_PACK"] = str(model_pack_build_path)
        with initialize_config_dir(config_dir=str(model_pack_build_path.joinpath("conf"))):
            cfg = compose(
                config_name="config",
                overrides=[],
            )
            ModelPackBuilder.build_caches(cfg)
            if run_consistency_checks:
                check_annotation_consistency(cfg)
            if run_acceptance_tests:
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
        parser_dict = instantiate(cfg.ontology_parser)
        curations_path = (
            Path(os.environ["KAZU_MODEL_PACK"])
            .joinpath("ontologies")
            .joinpath("curations")
            .joinpath("explosion_whitelist.jsonl")
        )
        explosion_path = Path(os.environ["KAZU_MODEL_PACK"]).joinpath("spacy_pipeline")
        assemble_pipeline.main(
            parsers=parser_dict.values(),
            curated_list=curations_path,
            output_dir=explosion_path,
        )
        load_steps(cfg)


def build_custom_pack_params(arguments) -> Optional[List[Tuple[Path, bool, bool]]]:
    if arguments.custom_model_pack_paths is None:
        return None
    return [
        (
            path,
            args.custom_packs_to_use_base_configuration[i],
            args.custom_packs_to_use_base_resources[i],
        )
        for i, path in enumerate(args.custom_model_pack_paths)
    ]


def _validate_arguments(arguments):
    if arguments.custom_model_pack_paths is not None:
        if not (
            len(arguments.custom_model_pack_paths)
            == len(arguments.custom_packs_to_use_base_configuration)
            == len(arguments.custom_packs_to_use_base_resources)
        ):
            raise ModelPackBuildError(
                "the number of arguments to --custom_model_pack_paths, --custom_packs_to_use_base_configuration"
                " and --custom_packs_to_use_base_resources"
                " should all be the same length"
            )
    if (
        arguments.build_base_model_pack_from_path is not None
        and arguments.base_configuration_path is None
    ):
        raise ModelPackBuildError(
            "--build_base_model_pack was specified, but --default_configuration_base_path was not "
        )
    if arguments.build_base_model_pack_from_path is None and (
        any(arguments.custom_packs_to_use_base_configuration)
        or any(arguments.custom_packs_to_use_base_resources)
    ):
        raise ModelPackBuildError(
            "a custom pack has been configured to use the base resources, but the --build_"
            "base_model_pack_from_path argument was not supplied"
        )


if __name__ == "__main__":
    description = """Build and test model packs. This script takes a list of model
pack directories, and creates zips of the model pack with any generated caches that are required. This script
requires each model pack directory to contain the following:

1) a conf dir (Hydra configuration for the model pack), containing
 a) configurations for a pipeline to be created
 b) a LabelStudioManager configuration to run acceptance tests
2) an explosion_whitelist.jsonl in the path <model_pack_root>/ontologies/curations/explosion_whitelist.jsonl
  in order for an ExplosionStringMatchingStep to be created
3) an acceptance_criteria.json in the path <model_pack_root>/acceptance_criteria.json in order for the
  acceptance tests to run. This file should specify the thresholds per NER class/Ontology for NER/linking
  respectively (see the provided model pack for an example)

In addition, if run_acceptance_tests or run_consistency_checks are specified, an instance of Label Studio
should be accessible on the network, in order for the acceptance test to be able to retrieve the annotations.
Access parameters for Label Studio are taken from the LabelStudioManager config in the model pack path.
"""

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--build_base_model_pack_from_path",
        type=Path,
        required=False,
        help="should the base model pack be built? If so, specify path "
        "to base model pack resources ",
    )
    parser.add_argument(
        "--base_configuration_path",
        type=Path,
        required=False,
        help="""Path to the base configuration to use if a model pack requires it (generally
        this should be <kazu repo root>/kazu/conf)""",
    )
    parser.add_argument(
        "--custom_model_pack_paths",
        type=Path,
        nargs="+",
        required=False,
        help="paths to model packs to build",
    )
    parser.add_argument(
        "--custom_packs_to_use_base_configuration",
        type=lambda x: bool(ast.literal_eval(x)),
        nargs="+",
        required=False,
        help="""should the custom model pack conf dir use the kazu base configuration, before applying any custom overrides?""",
    )
    parser.add_argument(
        "--custom_packs_to_use_base_resources",
        type=lambda x: bool(ast.literal_eval(x)),
        nargs="+",
        required=False,
        help="""should the custom model pack conf dir use the kazu base resources, before applying any custom overrides?""",
    )
    parser.add_argument(
        "--run_acceptance_tests",
        action="store_true",
        help="should the acceptance tests be run?",
    )
    parser.add_argument(
        "--run_consistency_checks",
        action="store_true",
        help="should consistency checks be run against the annotated corpus, to highlight potential annotation errors?",
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

    args = parser.parse_args()

    _validate_arguments(args)

    custom_model_pack_params = build_custom_pack_params(args)

    ModelPackBuilder.build_all_model_packs(
        maybe_base_model_pack_path=args.build_base_model_pack_from_path,
        maybe_base_configuration_path=args.base_configuration_path,
        custom_model_pack_params=custom_model_pack_params,
        zip_pack=args.zip_model_pack,
        run_consistency_checks=args.run_consistency_checks,
        run_acceptance_tests=args.run_acceptance_tests,
        output_dir=args.model_pack_output_path,
    )
