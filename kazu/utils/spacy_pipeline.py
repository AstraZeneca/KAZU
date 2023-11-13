import functools
import logging
import os
from collections import defaultdict
from collections.abc import Callable, Iterable
from copy import deepcopy
from string import ascii_lowercase
from typing import Union, Any, overload, Literal

import spacy
from kazu.utils.utils import Singleton
from spacy.lang.char_classes import (
    LIST_ELLIPSES,
    LIST_ICONS,
    CONCAT_QUOTES,
    ALPHA_LOWER,
    ALPHA_UPPER,
    ALPHA,
    HYPHENS,
)
from spacy.lang.en import English, EnglishDefaults
from spacy.lang.en.tokenizer_exceptions import TOKENIZER_EXCEPTIONS
from spacy.language import Language
from spacy.tokens import Doc

logger = logging.getLogger(__name__)

SPACY_DEFAULT_INFIXES = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        # note: this will get removed below
        r"(?<=[{a}0-9])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)

# We remove the hypen infix pattern because splitting by hyphens causes problems like
# splitting the company name 'ALK-Abello' into ['ALK', '-', 'Abello']
# and then potentially recognising ALK as a gene.

# A spacy PR that modified this hyphen pattern specifically mentions biomedical texts
# as being potentially problematic for this rule:
# https://github.com/explosion/spaCy/pull/5770#issuecomment-659389160
DEFAULT_INFIXES_MINUS_HYPHENS = SPACY_DEFAULT_INFIXES[:-2] + SPACY_DEFAULT_INFIXES[-1:]

# There are exceptions saying not to tokenize things like 'a.' and 'b.' - presumably
# to handle lists elements with letters like this. That isn't relevant for how we're
# using spacy to do NER + linking, and it breaks some cases for us - when an entity
# like 'Haemophilia A' is at the end of a sentence, so we get a token like 'A.', which
# then doesn't get split, so we don't recognise it, unless we remove these exceptions.
TOKENIZER_EXCEPTIONS_MINUS_SINGLE_LETTER = deepcopy(TOKENIZER_EXCEPTIONS)
for letter in ascii_lowercase:
    TOKENIZER_EXCEPTIONS_MINUS_SINGLE_LETTER.pop(letter + ".")


class KazuCustomEnglishDefaults(EnglishDefaults):
    tokenizer_exceptions = TOKENIZER_EXCEPTIONS_MINUS_SINGLE_LETTER
    infixes = [r"\(", "/"] + DEFAULT_INFIXES_MINUS_HYPHENS
    # related to the above, '.' isn't picked up as a suffix when preceded by a
    # single uppercase character
    suffixes = EnglishDefaults.suffixes + [  # type:ignore[operator] # because mypy
        r"(?<=\b[{au}])\.".format(
            au=ALPHA_UPPER
        )  # doesn't know that EnglishDefaults.suffixes is always a list[str]
    ]


@spacy.registry.languages("kazu_custom_en")
class KazuCustomEnglish(English):
    lang = "kazu_custom_en"
    Defaults = KazuCustomEnglishDefaults


BASIC_PIPELINE_NAME = "basic"


def basic_spacy_pipeline() -> Language:
    """A basic spaCy pipeline with a sentence splitter and a customised tokenizer."""
    nlp = spacy.blank("kazu_custom_en")
    nlp.add_pipe("sentencizer")
    return nlp


class SpacyPipelines(metaclass=Singleton):
    """Wraps spaCy pipelines into a singleton, so multiple can be accessed from
    different locations without additional memory overhead.

    In addition, due to a
    `known memory issue <https://github.com/explosion/spaCy/discussions/9362>`_,
    we reload each pipeline after a certain number of calls, namely :attr:`~.reload_at`.

    .. note::
        In order for the Garbage Collector to remove old spaCy Vocab objects, users
        should ensure that `spacy.Doc <https://spacy.io/api/doc>`_ objects are de-referenced as soon as possible
        (i.e. that you don't keep the results of :meth:`~.process_single` or :meth:`~.process_batch` around for
        a long time).
    """

    def __init__(self):
        self.reload_at = int(os.getenv("KAZU_SPACY_RELOAD_INTERVAL", 1000))
        """The interval (number of calls) at which spaCy models are reloaded.

        Normally set within the kazu model pack config using the environment variable
        ``KAZU_SPACY_RELOAD_INTERVAL``, but if this isn't set (either using the config
        or as a normal environment variable), this defaults to 1000.

        .. note::
            As this class is a singleton, modifying this will change the reload value for all
            spaCy pipelines (i.e. globally).

        :type: int
        """
        self.name_to_func: dict[str, Callable[[], Language]] = {}
        self.name_to_reload_callbacks: defaultdict[str, list[Callable[[], None]]] = defaultdict(
            list
        )
        self.call_counter: defaultdict[str, int] = defaultdict(lambda: 0)
        self.name_to_model: dict[str, Language] = {}

    @staticmethod
    def add_from_path(name: str, path: str) -> None:
        """Add a spaCy model from a path.

        Convenience function to call :meth:`~.add_from_func` with a
        wrapped version of
        `spacy.load <https://spacy.io/api/top-level#spacy.load>`_
        with the relevant path argument.
        """
        SpacyPipelines.add_from_func(name=name, func=functools.partial(spacy.load, path))

    @staticmethod
    def add_from_func(name: str, func: Callable[[], Language]) -> None:
        """Add a spaCy model from a callable."""

        instance = SpacyPipelines()
        if name in instance.name_to_func:
            logger.info("The spaCy pipeline key %s is already loaded.", name)
        else:
            instance.name_to_func[name] = func
            instance.name_to_model[name] = func()

    @staticmethod
    def add_reload_callback_func(name: str, func: Callable[[], None]) -> None:
        """Add a callback when a model is reloaded.

        If using spaCy components outside the context of a
        `Language <https://spacy.io/api/language>`_, these will also need to be reloaded when the
        underlying model is reloaded. This can be done by providing a zero
        argument, None return type callable. If you need to modify a field of an object with a
        callback, we recommend the use of :func:`functools.partial` to build the callback function.

        :param name:
        :param func:
        :return:
        """
        instance = SpacyPipelines()
        instance.name_to_reload_callbacks[name].append(func)

    @staticmethod
    def get_model(name: str) -> Language:
        """Get the underlying `Language <https://spacy.io/api/language>`_ for a
        given model key."""
        return SpacyPipelines().name_to_model[name]

    @overload
    def process_batch(
        self,
        texts: Iterable[tuple[Union[str, Doc], Any]],
        model_name: str,
        as_tuples: Literal[True],
        **kwargs: Any,
    ) -> Iterable[tuple[Doc, Any]]:
        ...

    @overload
    def process_batch(
        self,
        texts: Iterable[Union[str, Doc]],
        model_name: str,
        as_tuples: Literal[False] = False,
        **kwargs: Any,
    ) -> Iterable[Doc]:
        ...

    def process_batch(
        self,
        texts,
        model_name,
        as_tuples=False,
        **kwargs,
    ):
        """Process an iterable of `spacy.Doc`_ or strings with a given spaCy model.

        :param texts: either an iterable of 'texts'
            (`spacy.Doc`_\\ s or :class:`str`\\ s)
            if ``as_tuples=False`` (the default),
            or an iterable of length 2 tuples ``(text, context)`` where context
            is an arbitrary python object that is provided back in the output
            alongside its matching text.
        :param model_name: spaCy model to process texts with
        :param as_tuples: If set to ``True`` (the default is ``False``), texts are paired with
            a 'context' inside a tuple both in input and output. See the description of the
            texts argument and the output, and read the type information.
        :param kwargs: passed to `Language.pipe <https://spacy.io/api/language#pipe>`_\\ .
        :return: Iterable of `spacy.Doc`_\\ s if ``as_tuples=False``
            (the default).
            If ``as_tuples=True``, the result is an Iterable of ``(doc, context)`` tuples.

        .. _spacy.Doc: https://spacy.io/api/doc
        """
        for result in self.name_to_model[model_name].pipe(
            texts=texts, as_tuples=as_tuples, **kwargs
        ):
            yield result
            self.call_counter[model_name] += 1

        self._reload_if_required(model_name)

    def process_single(self, text: Union[str, Doc], model_name: str, **kwargs: Any) -> Doc:
        """Process a single `Doc <https://spacy.io/api/doc>`_ or text with a
        given spaCy model.

        :param text:
        :param model_name:
        :param kwargs: passed to the call method of `Language <https://spacy.io/api/language>`_
        :return:
        """
        res = self.name_to_model[model_name](text, **kwargs)
        self.call_counter[model_name] += 1
        self._reload_if_required(model_name)
        return res

    def _reload_if_required(
        self,
        model_name: str,
    ) -> None:
        if self.call_counter[model_name] >= self.reload_at:
            logger.info(
                "max spaCy calls (%s) exceeded for %s, "
                "see https://github.com/explosion/spaCy/discussions/9362 for more info",
                self.reload_at,
                model_name,
            )
            self.reload_model(model_name)
            self.call_counter[model_name] = 0

    def reload_model(self, model_name: str) -> None:
        """Reload a model, clearing the spaCy vocab."""
        func = self.name_to_func[model_name]
        logger.info("The model will be reloaded from %s.", func)
        self.name_to_model[model_name] = func()
        for callback in self.name_to_reload_callbacks[model_name]:
            callback()
