from typing import List, Protocol, Tuple, TypedDict

from kazu.data.data import Document


class StepMetadata(TypedDict):
    has_run: bool


class Step(Protocol):
    """
    abstract class for components. Describes signature of __call__ for all subclasses
    concrete implementations should implement the _run() method
    """

    @classmethod
    def namespace(cls) -> str:
        """
        the namespace is a piece of metadata to describe the step, and is used in various places.
        defaults to  cls.__name__
        """
        return cls.__name__

    def __call__(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        the main method to implement. Takes a list of docs, and returns a tuple where the first element
        is the succeeded docs, the second are the docs that failed to process. The logic of determining
        these two lists is the responsibility of the implementation.

        :param docs:
        :return:
        """
        raise NotImplementedError()
