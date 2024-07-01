import os

import streamlit as st
from hydra import initialize_config_dir, compose
from omegaconf import DictConfig


@st.cache_resource(show_spinner="Loading configuration")
def load_config() -> DictConfig:
    """Loads the configuration for the ResourceManager instance.

    :return:
    """
    conf_dir = os.environ["KAZU_MODEL_PACK"] + "/conf"
    from kazu.utils.constants import HYDRA_VERSION_BASE

    with initialize_config_dir(version_base=HYDRA_VERSION_BASE, config_dir=str(conf_dir)):
        cfg = compose(config_name="config", overrides=[])
        return cfg
