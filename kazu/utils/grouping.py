from itertools import groupby
from typing import Callable, Iterable, Tuple, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

Item = TypeVar("Item")
Key = TypeVar("Key", bound="SupportsRichComparison")


def sort_then_group(
    items: Iterable[Item], key_func: Callable[[Item], Key]
) -> Iterable[Tuple[Key, Iterable[Item]]]:
    sorted_items = sorted(items, key=key_func)
    yield from groupby(sorted_items, key=key_func)
