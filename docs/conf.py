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
import sys
from typing import Any

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
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.linkcode",
    "myst_parser",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv", "_changelog.d"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_logo = "kazu_logo.png"

# we use this to get module and class members documented as part of the autosummary-generated pages
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# put type hints in the description rather than the function signature, where it all ends up on one line and looks very ugly
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# ignore kazu for sorting modules, since everything starts with this
modindex_common_prefix = ["kazu."]

# used by both linkcode_resolve and furo's edit button
remote_base_url = "https://github.com/AstraZeneca/KAZU"


# config for edit button
html_theme_options = {
    "source_repository": remote_base_url,
    "source_branch": "main",
    "source_directory": "docs/",
    "sidebar_hide_name": True,
}


# set these variables everywhere to avoid repeating
# to decide whether we skip tests in docs
doctest_global_setup = """
import os

kazu_model_pack_missing = os.environ.get("KAZU_MODEL_PACK") is None
"""

# this means we don't try to generate docs for the conf or tests modules
# For me, this didn't work as expected with the 'autodoc-skip-member'
# event and a configured handler:
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-skip-member
# or using exclude_patterns
autodoc_mock_imports = ["kazu.conf", "kazu.tests"]

# groups 'members' together by type, e.g. attributes, methods etc.
autodoc_member_order = "groupwise"

intersphinx_mapping = {
    "transformers": ("https://huggingface.co/docs/transformers/main/en/", None),
    "pytorch_lightning": ("https://pytorch-lightning.readthedocs.io/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "python": ("https://docs.python.org/3", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}


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

    obj: Any = submod
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

    return f"{remote_base_url}/blob/main/kazu/{fn_filepath}{linespec}"


# raise failed links as warning, except for the ignored ones below
nitpicky = True

nitpick_ignore = [
    ###### Checked that there's no obvious solution #####
    # sphinx doesn't seem to handle typeVars here, see https://github.com/sphinx-doc/sphinx/issues/10974
    ("py:class", "kazu.utils.grouping.Item"),
    ("py:class", "kazu.utils.grouping.Key"),
    ("py:class", "kazu.steps.step.Self"),
    # to work, this has to be # transformers.BatchEncoding, but this isn't
    # the location of the 'live object' after importing.
    # This is an issue related to https://github.com/sphinx-doc/sphinx/issues/4826
    # I want to try and resolve this via https://github.com/theislab/scanpydoc/issues/64
    # this may affect other ignored errors below here too, I haven't fully chcked
    ("py:class", "transformers.tokenization_utils_base.BatchEncoding"),
    # This doesn't seem to have a py:class reference in the Python docs, but there is this in std:doc :
    # dom-document-objects                     Document Objects                        : library/xml.dom.html#dom-document-objects
    # again, with type hints though, overriding would be a pain
    ("py:class", "xml.dom.minidom.Document"),
    # suspect related to the above
    ("py:class", "xml.dom.minidom.Element"),
    ##### Not Checked, may be trivial to fix (just ignoring so we can avoid introducing new problems going forward) ######
    ("py:class", "JWTAuthenticationBackend"),
    ("py:class", "Module"),
    ("py:class", "batch"),
    (
        "py:class",
        "pytorch_lightning.core.module.LightningModule.forward",
    ),
    ("py:class", "omegaconf.dictconfig.DictConfig"),
    ("py:class", "omegaconf.listconfig.ListConfig"),
    ("py:class", "pydantic.main.BaseModel"),
    ("py:class", "rdflib.graph.Graph"),
    ("py:class", "rdflib.paths.Path"),
    ("py:class", "rdflib.term.Node"),
    ("py:class", "re.Pattern"),
    ("py:class", "spacy.language.Language"),
    ("py:class", "spacy.matcher.phrasematcher.PhraseMatcher"),
    ("py:class", "spacy.tokens.doc.Doc"),
    ("py:class", "spacy.tokens.span.Span"),
    ("py:class", "stanza.pipeline.core.Pipeline"),
    ("py:class", "starlette.authentication.AuthCredentials"),
    ("py:class", "starlette.authentication.AuthenticationBackend"),
    ("py:class", "starlette.authentication.BaseUser"),
    ("py:class", "starlette.middleware.base.BaseHTTPMiddleware"),
    ("py:class", "starlette.requests.HTTPConnection"),
    ("py:class", "starlette.requests.Request"),
    ("py:class", "starlette.responses.Response"),
    ("py:class", "transformers.data.data_collator.DataCollatorWithPadding"),
    ("py:class", "transformers.data.processors.utils.DataProcessor"),
    ("py:class", "transformers.data.processors.utils.InputExample"),
    ("py:class", "transformers.modeling_utils.PreTrainedModel"),
    ("py:class", "transformers.models.bert.modeling_bert.BertPreTrainedModel"),
    ("py:class", "transformers.tokenization_utils.PreTrainedTokenizer"),
    ("py:class", "transformers.tokenization_utils_base.PreTrainedTokenizerBase"),
    ("py:class", "transformers.tokenization_utils_fast.PreTrainedTokenizerFast"),
    ("py:class", "transformers.utils.generic.PaddingStrategy"),
]
