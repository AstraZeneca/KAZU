"""A sphinx extension to allow specifying where to look for cross reference links, particularly useful for type hint linking.

This particularly helps when using external types in a type hint, that you expect to be resolvable with intersphinx, but intersphinx
doesn't find the class documented where it expects it to be.

Effort will be saved if this is configured after intersphinx (although the behaviour will not change).

In essense, we allow configuring a mapping of 'where Sphinx (and intersphinx) expect to find an object' to 'where it will actually
find it'.

The problem in more detail: when Sphinx tries to create references for type hints, it ultimately (usually) uses the __qualname__
property of the 'live' object in the type hint. For example, if you do `from transformers import BatchEncoding`
and then use `BatchEncoding` as a type hint, Sphinx will try to look up `transformers.tokenization_utils_base.BatchEncoding`
in the object inventory. With intersphinx however, transformers has this object as `transformers.BatchEncoding` in its object
inventory, so the link fails.

I was inspired to try to work around this using Sphinx extension hooks by Scanpydoc https://github.com/theislab/scanpydoc
(see particularly https://github.com/theislab/scanpydoc/blob/a33f2803a1cd1f086d73da5707362003f21b920b/scanpydoc/elegant_typehints/__init__.py#L97
and
https://github.com/theislab/scanpydoc/blob/a33f2803a1cd1f086d73da5707362003f21b920b/scanpydoc/elegant_typehints/autodoc_patch.py#L21)
as well as this comment https://github.com/sphinx-doc/sphinx/issues/10151#issuecomment-1185794607 .

However, because the problem only materializes for kazu for intersphinx references in type hints, which is slightly different
to the two inspirations above, the implementation is not related.
"""

from typing import Dict, Any, Optional

from docutils.nodes import Element, TextElement
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext.intersphinx import resolve_reference_detect_inventory


def override_cross_reference(
    app: Sphinx, env: BuildEnvironment, node: pending_xref, contnode: TextElement
) -> Optional[Element]:
    """Override type hints that aren't registered in the object inventory in the place Sphinx expects.

    This function works by plugging in to the 'missing-reference' hook that Sphinx provided, which fires when a
    cross-reference has failed to resolve. For more info, see
    https://www.sphinx-doc.org/en/master/extdev/appapi.html#event-missing-reference .

    The function overrides the target of the cross-reference and then cals intersphinx's normal
    cross-reference resolution mechanism.
    """
    xref_override = app.config.cross_reference_override_mapping.get(node["reftarget"])
    if xref_override is not None:
        node["reftarget"] = xref_override
        return resolve_reference_detect_inventory(env, node, contnode)
    else:
        # no override, don't try to resolve
        return None


def setup(app: Sphinx) -> Dict[str, Any]:
    # some light experimentation suggests 'env' is necessary for the rebuild over just 'html'
    app.add_config_value(
        "cross_reference_override_mapping", default={}, rebuild="env", types=[Dict[str, str]]
    )
    app.connect("missing-reference", override_cross_reference)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
