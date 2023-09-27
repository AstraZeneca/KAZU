from itertools import groupby
from typing import TypeVar, TYPE_CHECKING
from collections.abc import Iterable, Callable

# see https://github.com/python/typeshed/tree/master/stdlib/_typeshed
# In short, this is needed as
# the _typeshed package and its types do not exist at runtime
if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

Item = TypeVar("Item")
"""The type of an element in the iterable being grouped."""
# we need to use Supports RichComparison here otherwise mypy complains
# about the 'key' argument to sorted
Key = TypeVar("Key", bound="SupportsRichComparison")
"""The type of the sort key provided by the key_func.

Bound to 'SupportsRichComparison' from
`_typeshed <https://github.com/python/typeshed/tree/main/stdlib/_typeshed>`_
as the keys must support comparison in order to be sorted using :func:`sorted`\\ .
"""


def sort_then_group(
    items: Iterable[Item], key_func: Callable[[Item], Key]
) -> Iterable[tuple[Key, Iterable[Item]]]:
    """.. without the override below, it fails to find Item and Key

    ..
      This is due to an issue with Sphinx handling builtins - see
      https://github.com/sphinx-doc/sphinx/issues/11571
      as it used to work while using typing.Callable and typing.Iterable
      instead of the collections.abc versions. Switching to show JsonDictType
      instead as it's simpler to write this override, plus I think more
      readable for users.
      An alternative would be to attempt to use Sphinx's autodoc_type_aliases
      https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_type_aliases
      but this requires doing ``from __future__ import annotations`` which could
      break pydantic stuff and have wider codebase implications, so this would
      be a potentially larger piece of work for not much gain.

    :param items:
    :type items: ~collections.abc.Iterable[~kazu.utils.grouping.Item]
    :param key_func:
    :type key_func: ~collections.abc.Callable[[~kazu.utils.grouping.Item], ~kazu.utils.grouping.Key]
    :rtype: ~collections.abc.Iterable[tuple[~kazu.utils.grouping.Key], ~collections.abc.Iterable[~kazu.utils.grouping.Item]]
    """
    sorted_items = sorted(items, key=key_func)
    yield from groupby(sorted_items, key=key_func)
