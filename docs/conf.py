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
from typing import Any, Union

import kazu

sys.path.insert(0, os.path.abspath("../kazu"))
sys.path.append(os.path.abspath("./_ext"))


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
    # custom extension, see docs/_ext.
    # saves effort by adding after intersphinx.
    "cross_reference_override",
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv", "_changelog.d", "_ext"]

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
    "special-members": "__call__",
}

# put type hints in the description rather than the function signature, where it all ends up on one line and looks very ugly
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# see https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-python_use_unqualified_type_names
# "If true, suppress the module name of the python reference if it can be resolved."
# We added this because Sphinx gives long, 'fully qualified' versions of the type with the module name
# when using builtin generics instead of those from typing
# (see https://github.com/sphinx-doc/sphinx/issues/11571)
# which makes the docs much less readable in some places.
# Turning on this option 'cancels this out', and has almost no other
# effect other than changing how things are italicised in some cases - I
# don't think the new style of italicising is particularly worse (if
# anything, it seems a little more consistent to me) so this seems like a
# good outcome for now. We could consider turning this option off in
# future if Sphinx resolves the issue above, but even in that case, it's
# not necessarily causing any harm.
python_use_unqualified_type_names = True

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
    "pytorch_lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "python": ("https://docs.python.org/3", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "diskcache": ("https://grantjenks.com/docs/diskcache/", None),
    "rdflib": ("https://rdflib.readthedocs.io/en/stable/", None),
    "pymongo": ("https://pymongo.readthedocs.io/en/stable/", None),
    # pymongo includes bson
    "bson": ("https://pymongo.readthedocs.io/en/stable/", None),
    "requests": ("https://requests.readthedocs.io/en/latest/", None),
    "urllib3": ("https://urllib3.readthedocs.io/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}


# this function is modified from the corresponding one in pandas, which in turn is modified from numpy
# both licenses are included in the docs folder, as PANDAS_LICENSE.txt and NUMPY_LICENSE.txt
def linkcode_resolve(domain: str, info: dict[str, Any]) -> Union[str, None]:
    """Determine the URL corresponding to Python object."""
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


# for custom cross_reference_override extension
cross_reference_override_mapping = {
    # these are cases where the location in the intersphinx object inventory doesn't match the __qualname__ on the live object
    # which is how Sphinx tries to look these up.
    "transformers.tokenization_utils_base.BatchEncoding": "transformers.BatchEncoding",
    "transformers.data.processors.utils.InputExample": "transformers.InputExample",
    "transformers.data.data_collator.DataCollatorWithPadding": "transformers.DataCollatorWithPadding",
    "transformers.data.processors.utils.DataProcessor": "transformers.DataProcessor",
    "transformers.modeling_utils.PreTrainedModel": "transformers.PreTrainedModel",
    "transformers.tokenization_utils.PreTrainedTokenizer": "transformers.PreTrainedTokenizer",
    "transformers.tokenization_utils_base.PreTrainedTokenizerBase": "transformers.PreTrainedTokenizerBase",
    "transformers.tokenization_utils_fast.PreTrainedTokenizerFast": "transformers.PreTrainedTokenizerFast",
    "transformers.utils.generic.PaddingStrategy": "transformers.utils.PaddingStrategy",
    "lightning_fabric.plugins.io.checkpoint_io.CheckpointIO": "lightning.pytorch.plugins.io.CheckpointIO",
    "pytorch_lightning.core.module.LightningModule": "lightning.pytorch.core.LightningModule",
    "pytorch_lightning.trainer.trainer.Trainer": "lightning.pytorch.trainer.trainer.Trainer",
    "urllib3.util.retry.Retry": "urllib3.util.Retry",
    "scipy.sparse._csr.csr_matrix": "scipy.sparse.csr_matrix",
}


nitpick_ignore = [
    # this doesn't exist anymore in lightning 2.0, it becomes on_validation_epoch_end, and there's some migration work for changing to it
    ("py:meth", "pytorch_lightning.core.LightningModule.validation_epoch_end"),
    # this doesn't appear to have an entry in the transformers docs for some reason.
    ("py:class", "transformers.models.bert.modeling_bert.BertPreTrainedModel"),
    # the kazu.utils.grouping.Key TypeVar tries to generate this automatically.
    # Sphinx doesn't find it because the class is in _typeshed, which doesn't exist at runtime.
    # We link to _typeshed docs from the docstring anyway, so this is fine for the user.
    ("py:class", "SupportsRichComparison"),
    # This isn't a class in the Python docs, there's a :ref: re-objects , but it's
    # used within a Union[str, re.Pattern], so a custom :type: override doesn't work nicely
    ("py:class", "re.Pattern"),
    # we can access omegaconf with intersphinx using https://omegaconf.readthedocs.io/en/latest/objects.inv
    # but their docs don't include an entry for DictConfig or ListConfig!
    ("py:class", "omegaconf.dictconfig.DictConfig"),
    ("py:class", "omegaconf.listconfig.ListConfig"),
    # spacy don't use Sphinx, they use a custom docs build process - they don't appear to have an objects.inv
    # we could go back and link at least some of these 'manually' with a :rtype: or :type: annotation
    ("py:class", "spacy.language.Language"),
    ("py:class", "spacy.matcher.phrasematcher.PhraseMatcher"),
    ("py:class", "spacy.tokens.doc.Doc"),
    ("py:class", "spacy.tokens.span.Span"),
    ("py:class", "spacy.tokens.token.Token"),
    ("py:class", "spacy.lang.en.English"),
    ("py:class", "spacy.lang.en.EnglishDefaults"),
    # stanza doesn't appear to build API docs (and the build process is unclear - they may not even use Sphinx at all)
    ("py:class", "stanza.pipeline.core.Pipeline"),
    # starlette doesn't appear to build API docs, and doesn't seem to use Sphinx.
    ("py:class", "starlette.authentication.AuthCredentials"),
    ("py:class", "starlette.authentication.AuthenticationBackend"),
    ("py:class", "starlette.authentication.BaseUser"),
    ("py:class", "starlette.middleware.base.BaseHTTPMiddleware"),
    ("py:class", "starlette.requests.HTTPConnection"),
    ("py:class", "starlette.requests.Request"),
    ("py:class", "starlette.responses.Response"),
    ("py:class", "starlette.responses.JSONResponse"),
    # pydantic uses mkdocs, not Sphinx, and doesn't seem to have full API docs
    ("py:class", "pydantic.main.BaseModel"),
    # ray does have sphinx docs (at https://docs.ray.io/en/latest/ , but we don't need them for anything else)
    # but it doesn't have a reference in its docs for ObjectRef (suprisingly)
    ("py:class", "ray._raylet.ObjectRef"),
    # regex doesn't seem to have API docs at all
    ("py:class", "_regex.Pattern"),
    ("py:class", "urllib3.util.retry.Retry"),
    ("py:class", "gliner.GLiNER"),
    # no sphinx for streamlit
    ("py:class", "streamlit.delta_generator.DeltaGenerator"),
    # vertexai
    ("py:class", "vertexai.generative_models._generative_models.SafetySetting"),
    ("py:class", "google.cloud.aiplatform_v1beta1.types.content.HarmCategory"),
    ("py:class", "google.cloud.aiplatform_v1beta1.types.content.SafetySetting.HarmBlockThreshold"),
]
