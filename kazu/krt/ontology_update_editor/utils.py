import functools
import inspect
from pathlib import Path
from typing import Any, Optional, cast

import pandas as pd
import streamlit as st
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict, OmegaConf

from kazu.data import OntologyStringResource
from kazu.krt.utils import resource_to_df
from kazu.ontology_preprocessing.base import OntologyParser
from kazu.ontology_preprocessing.curation_utils import dump_ontology_string_resources
from kazu.ontology_preprocessing.ontology_upgrade_report import OntologyUpgradeReport
from kazu.krt import load_config


class OntologyUpdateManager:
    """A class to manage the updating the raw ontology files of ontology parsers."""

    def __init__(self, parser_name: str):
        self.original_parser_cfg: DictConfig = self.name_to_parser_cfg()[parser_name]
        self.parser = instantiate(self.original_parser_cfg)
        self.new_parser_cfg: DictConfig = self.original_parser_cfg.copy()

    def get_downloader_args_and_types(self) -> dict[str, type]:
        """Get the types of the arguments of the
        :class:`kazu.ontology_preprocessing.downloads.OntologyDownloader` class."""
        return inspect.getfullargspec(
            self.parser.ontology_downloader.__class__.__init__
        ).annotations

    def get_downloader_args_and_vals(self) -> dict[str, Any]:
        """Get the arguments and their values of the configured
        :class:`kazu.ontology_preprocessing.downloads.OntologyDownloader` class."""
        configured_args_dict = dict(self.original_parser_cfg.ontology_downloader)
        # remove target as it is not an argument
        configured_args_dict.pop("_target_")
        return configured_args_dict

    def display_new_parser_config_as_yaml(self) -> str:
        return OmegaConf.to_yaml(self.new_parser_cfg)

    def display_new_downloader_config_as_yaml(self) -> str:
        return OmegaConf.to_yaml(self.new_parser_cfg.ontology_downloader)

    @staticmethod
    def name_to_parser_cfg() -> dict[str, DictConfig]:
        """Get the names of the parsers and their configurations.

        :return:
        """
        result = {}
        cfg = load_config()
        for parser_key, parser_config_dict in cfg.ontologies.parsers.items():
            result[parser_config_dict.name] = parser_config_dict
        return result

    def modify_downloader_cfg_with_new_args(self, kwargs: dict[str, Any]) -> None:
        """Modify the downloader configuration with the new arguments.

        :param kwargs:
        :return:
        """
        with open_dict(self.new_parser_cfg):
            for arg_name, arg_val in kwargs.items():
                self.new_parser_cfg.ontology_downloader[arg_name] = arg_val

    def modify_parser_cfg_with_new_args(self) -> None:
        """Modify the parser configuration with the new downloader configuration.

        :return:
        """
        parser = instantiate(self.new_parser_cfg)
        new_in_path = parser.ontology_downloader.download(parser.in_path, skip_download=True)
        with open_dict(self.new_parser_cfg):
            self.new_parser_cfg.data_origin = parser.ontology_downloader.version(parser.in_path)
            self.new_parser_cfg.in_path = str(new_in_path.absolute())

    def instantiate_parser_with_new_config(self) -> OntologyParser:
        """Instantiate a new parser with the new configuration.

        :return:
        """
        return cast(OntologyParser, instantiate(self.new_parser_cfg))

    @functools.cache
    def get_or_build_upgrade_report(self) -> OntologyUpgradeReport:
        """Get the upgrade report if it exists, otherwise build it.

        :return:
        """

        parser = self.instantiate_parser_with_new_config()
        return parser.upgrade_ontology_version()

    def download_ontology_with_new_config(self) -> None:
        """Download the ontology with the new configuration.

        :return:
        """
        parser = self.instantiate_parser_with_new_config()
        assert parser.ontology_downloader is not None
        parser.download_ontology()

    def obsolete_df(self) -> Optional[pd.DataFrame]:
        """Get the dataframe of the obsolete resources.

        :return:
        """
        return self._df_from_set(
            self.get_or_build_upgrade_report().obsolete_resources_after_upgrade
        )

    def _df_from_set(self, resources: set[OntologyStringResource]) -> Optional[pd.DataFrame]:
        if resources:
            df = pd.concat(resource_to_df(resource, True) for resource in resources)
            df["string_length"] = df["text"].apply(len)
            return df
        else:
            return None

    def novel_df(self) -> Optional[pd.DataFrame]:
        """Get the dataframe of the novel resources.

        :return:
        """
        return self._df_from_set(self.get_or_build_upgrade_report().new_resources_after_upgrade)

    def write_new_defaults_to_model_pack(self) -> Path:
        """Write the new default resources to the model pack.

        :return:
        """
        defaults_path = cast(Path, self.parser.ontology_auto_generated_resources_set_path)
        with st.spinner(f"updating defaults at {defaults_path}"):
            dump_ontology_string_resources(
                self.get_or_build_upgrade_report().new_version_auto_generated_resources_clean,
                defaults_path,
                force=True,
            )
        return defaults_path
