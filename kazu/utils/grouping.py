from itertools import groupby
from typing import Callable, Iterable, Tuple, TypeVar, TYPE_CHECKING

# see https://github.com/python/typeshed/tree/master/stdlib/_typeshed
# In short, this is needed as
# the _typeshed package and its types do not exist at runtime
if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

Item = TypeVar("Item")
# we need to use Supports RichComparison here otherwise mypy complains
# about the 'key' argument to sorted
Key = TypeVar("Key", bound="SupportsRichComparison")


def sort_then_group(
    items: Iterable[Item], key_func: Callable[[Item], Key]
) -> Iterable[Tuple[Key, Iterable[Item]]]:
    sorted_items = sorted(items, key=key_func)
    yield from groupby(sorted_items, key=key_func)
