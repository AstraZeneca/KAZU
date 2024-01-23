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
    items: Iterable[Item], key_func: Callable[[Item], Key], reverse: bool = False
) -> Iterable[tuple[Key, Iterable[Item]]]:
    sorted_items = sorted(items, key=key_func, reverse=reverse)
    yield from groupby(sorted_items, key=key_func)
