from abc import ABC
from typing import List, Tuple, Optional, TypedDict

from kazu.data.data import Document


class StepMetadata(TypedDict):
    has_run: bool


class Step(ABC):
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

    def __init__(self, depends_on: Optional[List[str]]):
        """
        :param depends_on: a list of step namespaces that this step expects. Note, this is not used by the step itself,
            but should be used via some step orchestration logic (e.g. Pipeline) to determine whether the step should
            run or not.
        """
        self.depends_on = depends_on if depends_on is not None else []

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        the main method to implement. Takes a list of docs, and returns a tuple where the first element
        is the succeeded docs, the second are the docs that failed to process. The logic of determining
        these two lists is the responsibility of the implementation.

        :param docs:
        :return:
        """
        raise NotImplementedError()

    def __call__(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        return self._run(docs)
