from functools import wraps
import traceback
from typing import Any, Callable, List, Protocol, Tuple, TypedDict, TypeVar

from kazu.data.data import Document, PROCESSING_EXCEPTION


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
        # Note: it would be nice for this to be a property, but you can't use @classmethod
        # and @property together in most versions of python (you could in 3.9 and 3.10), and
        # it being a classmethod is quite useful.
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


Self = TypeVar("Self")


def iterating_step(
    per_doc_callable: Callable[[Self, Document], Any]
) -> Callable[[Self, List[Document]], Tuple[List[Document], List[Document]]]:
    """Handle a list of :class:`~kazu.data.data.Document`\\ s and add error handling.

    Decorate a method that processes single :class:`~kazu.data.data.Document`\\ s with
    this, and then resulting function will then iterate over a list of
    :class:`~kazu.data.data.Document`\\ s, calling the original function, and error handling
    and providing a return value to match the return type of the ``__call__`` method of a
    :class:`Step`\\ .

    Note that this will only work for a method of a class, rather than a standalone function,
    as it expects to have to pass through 'self' as a parameter.

    :param per_doc_callable: A function that processes a single document, that you want to use as
        the :literal:`__call__` method of a :class:`Step`\\ . This must do its work by mutating the
        input document: the return value is ignored.
    """
    # note - this excludes __annotations__ as the returned function has a different type signature
    @wraps(per_doc_callable, assigned=("__module__", "__name__", "__qualname__", "__doc__"))
    def step_call(self: Self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        for doc in docs:
            try:
                per_doc_callable(self, doc)
            except Exception:
                doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                failed_docs.append(doc)
        return docs, failed_docs

    return step_call


def batch_step(
    batch_doc_callable: Callable[[Self, List[Document]], Any]
) -> Callable[[Self, List[Document]], Tuple[List[Document], List[Document]]]:
    """Add error handling to a method that processes batches of :class:`~kazu.data.data.Document`\\ s.

    Decorate a function that processes single :class:`~kazu.data.data.Document`\\ s with
    this, and then resulting function will then iterate over a list of
    :class:`~kazu.data.data.Document`\\ s, calling the original function, and error handling
    and providing a return value to match the return type of the ``__call__`` method of a
    :class:`Step`\\ .

    Note that this will only work for a method of a class, rather than a standalone function,
    as it expects to have to pass through 'self' as a parameter.

    :param batch_doc_callable: A function that processes a batch of documents, that you want to use
        as the :literal:`__call__` method of a :class:`Step`. This must do its work by mutating the
        input documents: the return value is ignored.
    """
    # note - this excludes __annotations__ as the returned function has a different type signature
    @wraps(batch_doc_callable, assigned=("__module__", "__name__", "__qualname__", "__doc__"))
    def step_call(self: Self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        try:
            batch_doc_callable(self, docs)
        except Exception:
            affected_doc_ids = [doc.idx for doc in docs]
            for doc in docs:
                message = (
                    f"batch failed: affected ids: {affected_doc_ids}\n" + traceback.format_exc()
                )
                doc.metadata[PROCESSING_EXCEPTION] = message
                failed_docs.append(doc)
        return docs, failed_docs

    return step_call
