import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from distutils.dir_util import copy_tree
from pathlib import Path
from typing import Tuple, List, Optional, Set, Dict, Iterable

from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from kazu.modelling.annotation.acceptance_test import (
    execute_full_pipeline_acceptance_test,
    check_annotation_consistency,
)
from kazu.modelling.ontology_matching import assemble_pipeline
from kazu.pipeline import load_steps_and_log_memory_usage
from kazu.utils.utils import Singleton
from omegaconf import DictConfig


@dataclass
class BuildConfiguration:
    """Dataclass that controls how a base model pack and config should be merged with a
    target model pack.

    :param use_base_config: should this model pack use the base config as a starting point?
    :param models: what model directories should this model pack include from the base model pack?
    :param curations: what entity classes should this model pack use from the curated list in the base model pack?
    :param ontologies: what ontologies should this model pack use from the base ontology pack?
        (elements iterables of path parts, e.g.  ('ontologies','cl.owl'))
    :param has_own_config: does this model pack have its own config dir? (if used with use_base_config
        these will override any config files from the base config)
    :param run_acceptance_tests: should acceptance tests be run?
    :param run_consistency_checks: should consistency checks be run on the gold standard?
    """

    use_base_config: bool
    models: List[str]
    curations: Set[str]
    ontologies: List[Iterable[str]]
    has_own_config: bool
    run_acceptance_tests: bool = False
    run_consistency_checks: bool = False

    def __post_init__(self):
        self.curations = set(self.curations)


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
    def resolve_curations(
        base_model_pack_path: Path,
        target_model_pack_path: Path,
        merge_config: BuildConfiguration,
    ) -> List[Dict]:
        """
        load curations from the base model pack based upon the required entity classes specified
        in the :class:`.MergeConfiguration`\\, then load any additional ones from the target
        model pack path. Finally, merge the two lists, warning on any inconsistencies that occur

        :param base_model_pack_path:
        :param target_model_pack_path:
        :param merge_config:
        :return:
        """

        base_curations = ModelPackBuilder.load_curations(
            base_model_pack_path, merge_config.curations
        )
        build_curations = ModelPackBuilder.load_curations(target_model_pack_path, None)
        final_curations = {}
        for k, v in base_curations.items():
            if k in build_curations:
                new_v = build_curations.pop(k)
                if v["action"] != new_v["action"] and new_v["action"] == "keep":
                    print(f"warning: previously dropped term is now being kept: {new_v}")
                if v["case_sensitive"] != new_v["case_sensitive"] and not new_v["case_sensitive"]:
                    print(f"warning: case sensitivity is now less strict for {new_v}")
                final_curations[k] = new_v
        final_curations.update(build_curations)
        return list(final_curations.values())

    @staticmethod
    def apply_merge_configurations(
        build_config: BuildConfiguration,
        maybe_base_model_pack_path: Optional[Path],
        maybe_base_configuration_path: Optional[Path],
        model_pack_build_path: Path,
        uncached_model_pack_path: Path,
    ):

        # copy the target pack to the target build dir
        copy_tree(str(uncached_model_pack_path), str(model_pack_build_path))
        # copy base config if required
        if build_config.use_base_config:
            if maybe_base_configuration_path is None:
                raise ModelPackBuildError(
                    f"merge config asked for base configuration path but none was provided: {build_config}"
                )
            model_pack_build_path.joinpath("conf").mkdir(exist_ok=True)
            copy_tree(
                str(maybe_base_configuration_path), str(model_pack_build_path.joinpath("conf"))
            )
        if build_config.has_own_config:
            # and copy target config to override required elements (if required)
            copy_tree(
                str(uncached_model_pack_path.joinpath("conf")),
                str(model_pack_build_path.joinpath("conf")),
            )

        if maybe_base_model_pack_path is None and (
            len(build_config.models) > 0
            or len(build_config.ontologies) > 0
            or len(build_config.curations) > 0
        ):
            raise ModelPackBuildError(
                f"merge config asked for base model pack path but none was provided: {build_config}"
            )
        assert isinstance(maybe_base_model_pack_path, Path)
        for model in build_config.models:
            model_source_path = maybe_base_model_pack_path.joinpath(model)
            target_dir = model_pack_build_path.joinpath(model_source_path.name)
            shutil.copytree(str(model_source_path), str(target_dir))
        for ontology_path_elements in build_config.ontologies:
            ontology_path = Path(os.path.join(maybe_base_model_pack_path, *ontology_path_elements))
            target_path = Path(os.path.join(model_pack_build_path, *ontology_path_elements))
            if ontology_path.is_dir():
                shutil.copytree(str(ontology_path), str(target_path))
            else:
                shutil.copy(ontology_path, target_path)

        curations = ModelPackBuilder.resolve_curations(
            maybe_base_model_pack_path, uncached_model_pack_path, build_config
        )
        curations_dir = model_pack_build_path.joinpath("ontologies").joinpath("curations")
        curations_dir.mkdir(parents=True, exist_ok=True)
        with open(curations_dir.joinpath("explosion_whitelist.jsonl"), "w") as f:
            for curation in curations:
                f.write(json.dumps(curation) + "\n")

    @staticmethod
    def load_curations(
        model_pack_path: Path, ent_class_filter: Optional[Set[str]]
    ) -> Dict[Tuple[str, str], Dict]:
        """
        load curations indexed by term and entity class

        :param model_pack_path:  path to root of model pack
        :param ent_class_filter: only return curations in this set of entity classes. If None, return all
        :return:
        """
        curations: Dict[Tuple[str, str], Dict] = {}
        target_path = (
            model_pack_path.joinpath("ontologies")
            .joinpath("curations")
            .joinpath("explosion_whitelist.jsonl")
        )
        if not target_path.exists():
            print(f"no curations found at {str(target_path)}")
            return curations

        with open(target_path) as f:
            for line in f:
                item: Dict = json.loads(line)
                if (
                    ent_class_filter is not None and item["entity_class"] in ent_class_filter
                ) or ent_class_filter is None:
                    curations[(item["term"], item["entity_class"])] = item

        return curations

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
        model_pack_paths: Optional[List[Path]],
        zip_pack: bool,
        output_dir: Path,
    ):
        """
        build multiple model packs

        :param maybe_base_model_pack_path: Path to the base model pack, if required
        :param maybe_base_configuration_path: Path to the base configuration, if required
        :param model_pack_paths: list of paths to model pack resources
        :param zip_pack: should the pack be zipped at the end?
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

        if model_pack_paths is not None:
            for model_pack_path in model_pack_paths:
                ModelPackBuilder.reset_singletons()
                print(f"building model pack at {model_pack_path}")
                ModelPackBuilder.process_model_pack_path(
                    maybe_base_model_pack_path=maybe_base_model_pack_path,
                    maybe_base_configuration_path=maybe_base_configuration_path,
                    kazu_version=kazu_version,
                    zip_pack=zip_pack,
                    uncached_model_pack_path=model_pack_path,
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
        maybe_base_model_pack_path: Optional[Path],
        maybe_base_configuration_path: Optional[Path],
        kazu_version: str,
        zip_pack: bool,
        uncached_model_pack_path: Path,
        build_dir: Path,
    ) -> Path:
        """
        run all configured options on a given model pack path

        :param maybe_base_model_pack_path: if this pack requires the base model pack, specify path
        :param maybe_base_configuration_path: if this pack requires the base configuration, specify path
        :param kazu_version: version of kazu used to generate model pack
        :param zip_pack: should model pack be zipped?
        :param uncached_model_pack_path: path to model pack to process
        :param build_dir: directory pack should be built in
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
        with initialize_config_dir(config_dir=str(model_pack_build_path.joinpath("conf"))):
            cfg = compose(
                config_name="config",
                overrides=[],
            )
            ModelPackBuilder.build_caches(cfg)
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
        curations_path = (
            Path(os.environ["KAZU_MODEL_PACK"])
            .joinpath("ontologies")
            .joinpath("curations")
            .joinpath("explosion_whitelist.jsonl")
        )
        explosion_path = Path(os.environ["KAZU_MODEL_PACK"]).joinpath("spacy_pipeline")
        assemble_pipeline.main(
            parsers=parsers,
            curated_list=curations_path,
            output_dir=explosion_path,
        )
        load_steps_and_log_memory_usage(cfg)


if __name__ == "__main__":
    description = """Build and test model packs. This script takes a list of model
pack directories, and creates model packs with a number of options. Depending on
how it is called, one or more of the following may be required:

1) a conf dir (Hydra configuration for the model pack), containing
 a) configurations for a pipeline to be created
 b) a LabelStudioManager configuration to run acceptance tests
2) an explosion_whitelist.jsonl in the path <model_pack_root>/ontologies/curations/explosion_whitelist.jsonl
  in order for an ExplosionStringMatchingStep to be created
3) an acceptance_criteria.json in the path <model_pack_root>/acceptance_criteria.json in order for the
  acceptance tests to run. This file should specify the thresholds per NER class/Ontology for NER/linking
  respectively (see the provided model pack for an example)
4) a merge config in the path <model_pack_root>/base_model_merge_config.json, which determines which elements
  of the base configuration and model pack should be used

In addition, if run_acceptance_tests or run_consistency_checks are specified, an instance of Label Studio
should be accessible on the network, in order for the acceptance test to be able to retrieve the annotations.
Access parameters for Label Studio are taken from the LabelStudioManager config in the model pack path.
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

    args = parser.parse_args()

    ModelPackBuilder.build_all_model_packs(
        maybe_base_model_pack_path=args.base_model_pack_path,
        maybe_base_configuration_path=args.base_configuration_path,
        model_pack_paths=args.model_packs_to_build,
        zip_pack=args.zip_model_pack,
        output_dir=args.model_pack_output_path,
    )
