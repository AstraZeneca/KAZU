from functools import wraps
import traceback
from typing import Any, Protocol, TypeVar
from collections.abc import Iterable, Callable

from kazu.data import Document, PROCESSING_EXCEPTION
from kazu.ontology_preprocessing.base import OntologyParser


class Step(Protocol):
    @classmethod
    def namespace(cls) -> str:
        """Metadata to name/describe the step, used in various places.

        Defaults to  ``cls.__name__``.
        """
        # Note: it would be nice for this to be a property, but you can't use @classmethod
        # and @property together in most versions of python (you could in 3.9 and 3.10), and
        # it being a classmethod is quite useful.
        return cls.__name__

    def __call__(self, docs: list[Document]) -> tuple[list[Document], list[Document]]:
        """Process documents and respond with processed and failed documents.

        Note that many steps will be decorated by :func:`~.kazu.steps.step.document_iterating_step`
        or :func:`~.kazu.steps.step.document_batch_step` which will modify the 'original'
        ``__call__`` function signature to match the expected signature for a step, as the
        decorators handle the exception/failed documents logic for you.

        :param docs:
        :return: The first element is all the provided docs (now modified by the processing), the
            second is the docs that failed to (fully) process correctly.
        """
        raise NotImplementedError


class ParserDependentStep(Step):
    """A step that depends on ontology parsers in any form.

    Steps that need information from parsers should subclass this class, in order for
    the internal databases to be correctly populated. Generally, these will be steps
    that have anything to do with Entity Linking.
    """

    def __init__(self, parsers: Iterable[OntologyParser]):
        """

        :param parsers: parsers that this step requires
        """
        for parser in parsers:
            parser.populate_databases(force=False)


Self = TypeVar("Self")
"""A TypeVar for the type of the class whose method is decorated with
:func:`~.kazu.steps.step.document_iterating_step` or
:func:`~.kazu.steps.step.document_batch_step`\\ ."""


def document_iterating_step(
    per_doc_callable: Callable[[Self, Document], Any]
) -> Callable[[Self, list[Document]], tuple[list[Document], list[Document]]]:
    """Handle a list of :class:`~kazu.data.Document`\\ s and add error handling.

    Use this to decorate a method that processes a single :class:`~kazu.data.Document`\\ .
    The resulting method will then iterate over a list of
    :class:`~kazu.data.Document`\\ s, calling the decorated function for each
    :class:`~kazu.data.Document`\\ . Errors are handled automatically and added to the
    ``PROCESSING_EXCEPTION`` metadata of documents, with failed docs returned as the second element
    of the return value, as expected by :meth:`Step.__call__`\\ .

    Generally speaking, it will save effort and repetition to decorate a :class:`Step` with either
    :func:`~.kazu.steps.step.document_iterating_step` or
    :func:`~.kazu.steps.step.document_batch_step`\\, rather than implementing the error handling in
    the :class:`Step` itself.

    Normally, :func:`~.kazu.steps.step.document_iterating_step` would be used in preference to
    :func:`~.kazu.steps.step.document_batch_step`\\, unless the method involves computation which
    is more efficient when run in a batch, such as inference with a transformer-based Machine
    Learning model, or using spaCy's `pipe <https://spacy.io/api/language/#pipe>`_ method.

    Note that this will only work for a method of a class, rather than a standalone function,
    as it expects to have to pass through 'self' as a parameter.

    :param per_doc_callable: A function that processes a single document, that you want to use as
        the :literal:`__call__` method of a :class:`Step`\\ . This must do its work by mutating the
        input document: the return value is ignored.
    :return:
    """
    # note - this excludes __annotations__ as the returned function has a different type signature
    @wraps(per_doc_callable, assigned=("__module__", "__name__", "__qualname__", "__doc__"))
    def step_call(self: Self, docs: list[Document]) -> tuple[list[Document], list[Document]]:
        failed_docs = []
        for doc in docs:
            try:
                per_doc_callable(self, doc)
            except Exception:
                doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                failed_docs.append(doc)
        return docs, failed_docs

    return step_call


def document_batch_step(
    batch_doc_callable: Callable[[Self, list[Document]], Any]
) -> Callable[[Self, list[Document]], tuple[list[Document], list[Document]]]:
    """Add error handling to a method that processes batches of
    :class:`~kazu.data.Document`\\ s.

    Use this to decorate a method that processes a batch of :class:`~kazu.data.Document`\\ s
    at a time. The resulting method will wrap a call to the decorated function with error handling
    which will add exceptions to the ``PROCESSING_EXCEPTION`` metadata of documents. Failed
    documents will be returned as the second element of the return value, as expected by
    :meth:`Step.__call__`\\ .

    Generally speaking, it will save effort and repetition to decorate a :class:`Step` with either
    :func:`~.kazu.steps.step.document_iterating_step` or
    :func:`~.kazu.steps.step.document_batch_step`\\, rather than implementing the error handling in
    the :class:`Step` itself.

    Normally, :func:`~.kazu.steps.step.document_iterating_step` would be used in preference to
    :func:`~.kazu.steps.step.document_batch_step`\\, unless the method involves computation which
    is more efficient when run in a batch, such as inference with a transformer-based Machine
    Learning model, or using spacy's `pipe <https://spacy.io/api/language/#pipe>`_ method.

    Note that this will only work for a method of a class, rather than a standalone function,
    as it expects to have to pass through 'self' as a parameter.

    :param batch_doc_callable: A function that processes a batch of documents, that you want to use
        as the :literal:`__call__` method of a :class:`Step`. This must do its work by mutating the
        input documents: the return value is ignored.
    :return:
    """
    # note - this excludes __annotations__ as the returned function has a different type signature
    @wraps(batch_doc_callable, assigned=("__module__", "__name__", "__qualname__", "__doc__"))
    def step_call(self: Self, docs: list[Document]) -> tuple[list[Document], list[Document]]:
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
