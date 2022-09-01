# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import inspect
import os
import subprocess
import sys

import kazu

sys.path.insert(0, os.path.abspath("../kazu"))


# -- Project information -----------------------------------------------------

project = "Kazu"
copyright = "2021, Korea University, AstraZeneca"
author = "Korea University, AstraZeneca"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "sphinx.ext.doctest", "sphinx.ext.linkcode", "myst_parser"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinxdoc"

# set these variables everywhere to avoid repeating
# to decide whether we skip tests in docs
doctest_global_setup = """
import os

kazu_config_missing = os.environ.get("KAZU_CONFIG_DIR") is None
kazu_model_pack_missing = os.environ.get("KAZU_MODEL_PACK") is None
"""


# this function is modified from the corresponding one in pandas, which in turn is modified from numpy
# both licenses are included in the docs folder, as PANDAS_LICENSE.txt and NUMPY_LICENSE.txt
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            # pandas has a warnings.catch_warnings context manager here
            # with a warnings.simplefilter to ignore FutureWarnings
            # to avoid noisy warnings over deprecated objects. We don't need
            # this for now, but maybe will want to bring it back in future.
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn_filepath = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn_filepath = None
    if not fn_filepath:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn_filepath = os.path.relpath(fn_filepath, start=os.path.dirname(kazu.__file__))

    # get the configured git remote. Get the url this way rather than hardcoding since
    # we don't want this to give internal links when we open source. We will need to consider
    # docs and hosting/linking more generally when open sourcing - this is just a basic precaution.
    remote_process = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"], capture_output=True, encoding="utf-8"
    )
    remote_base_url = remote_process.stdout.strip().removesuffix(".git")
    assert remote_base_url.startswith("https://github.com/")
    return f"{remote_base_url}/blob/main/kazu/{fn_filepath}{linespec}"
