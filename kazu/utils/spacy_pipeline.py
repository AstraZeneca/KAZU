import logging
import os
from collections import defaultdict
from collections.abc import Callable, Iterable
from copy import deepcopy
from string import ascii_lowercase
from typing import Union, Any

import spacy
from kazu.utils.utils import Singleton
from spacy.language import Language
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
    """A basic Spacy pipeline with a customised tokenizer and sentence
    splitter."""
    nlp = spacy.blank("kazu_custom_en")
    nlp.add_pipe("sentencizer")
    return nlp


class SpacyPipelines(metaclass=Singleton):
    """Wraps spacy pipelines into a singleton, so multiple can be accessed from
    different locations without additional memory overhead.

    In addition, due to a
    `known memory issue <https://github.com/explosion/spaCy/discussions/9362>`_
    , we reload each pipeline after a certain number of calls
    """

    def __init__(self):
        # because this is a singleton, we can't parameterise the reload variable in the constructor
        self._reload_at = int(os.getenv("KAZU_SPACY_RELOAD_INTERVAL", 1000))
        self.name_to_path_or_build_func: dict[str, Union[str, Callable[[], Language]]] = {}
        self.name_to_reload_callbacks: defaultdict[str, list[Callable[[], None]]] = defaultdict(
            list
        )
        self.call_counter: defaultdict[str, int] = defaultdict(lambda: 0)
        self.name_to_model: dict[str, Language] = {}

    @property
    def reload_at(self) -> int:
        instance = SpacyPipelines()
        return instance._reload_at

    @reload_at.setter
    def reload_at(self, value: int) -> None:
        """Change the interval at which spacy models are reloaded.

        Note, as this is a singleton, it will change the reload value for all
        spacy pipelines (i.e. globally)

        :param value: reload after this many calls
        :return:
        """

        instance = SpacyPipelines()
        instance._reload_at = value

    @staticmethod
    def add_from_path(name: str, path: str) -> None:
        """Add a spacy model from a path."""

        instance = SpacyPipelines()
        if name in instance.name_to_path_or_build_func:
            logger.info("The spacy pipeline key %s is already loaded.", name)
        else:
            instance.name_to_path_or_build_func[name] = path
        instance.name_to_model[name] = spacy.load(path)

    @staticmethod
    def add_from_func(name: str, func: Callable[[], Language]) -> None:
        """Add a spacy model from a callable."""

        instance = SpacyPipelines()
        if name in instance.name_to_path_or_build_func:
            logger.info("The spacy pipeline key %s is already loaded.", name)
        else:
            instance.name_to_path_or_build_func[name] = func
        instance.name_to_model[name] = func()

    @staticmethod
    def add_reload_callback_func(name: str, func: Callable[[], None]) -> None:
        """Add a callback when a model is reloaded.

        If using spacy components outside the context of a
        `Language <https://spacy.io/api/language>`_, these will also need to be reloaded when the
        underlying model is reloaded. This can be done by providing a zero
        argument, None return type callable.

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

    def process_batch(
        self,
        texts: Union[Iterable[Union[str, Doc]], Iterable[tuple[Union[str, Doc], Any]]],
        model_name: str,
        **kwargs: Any,
    ) -> Union[Iterable[Doc], Iterable[tuple[Doc, Any]]]:
        """Process an iterable of `Doc <https://spacy.io/api/doc>`_ or text
        with a given spacy model.

        :param texts:
        :param model_name:
        :param kwargs: passed to the pipe method of `Language <https://spacy.io/api/language>`_
        :return:
        """
        for result in self.name_to_model[model_name].pipe(texts=texts, **kwargs):
            yield result
            self.call_counter[model_name] += 1

        self._reload_if_required(model_name)

    def process_single(self, text: Union[str, Doc], model_name: str, **kwargs: Any) -> Doc:
        """Process a single `Doc <https://spacy.io/api/doc>`_ or text with a
        given spacy model.

        :param text:
        :param model_name:
        :param kwargs: passed to the call method of `Language <https://spacy.io/api/language>`_
        :return:
        """
        self.call_counter[model_name] += 1
        self._reload_if_required(model_name)
        return self.name_to_model[model_name](text, **kwargs)

    def _reload_if_required(
        self,
        model_name: str,
    ) -> None:
        if self.call_counter[model_name] >= self.reload_at:
            logger.info(
                "max spacy calls exceeded for %s, "
                "see https://github.com/explosion/spaCy/discussions/9362 for more info",
                model_name,
            )
            self.reload_model(model_name)
            self.call_counter[model_name] = 0

    def reload_model(self, model_name: str) -> None:
        """Reload a model, clearing the spacy vocab."""

        path_or_func = self.name_to_path_or_build_func[model_name]
        logger.info("The model will be reloaded from %s.", path_or_func)
        if isinstance(path_or_func, str):
            self.name_to_model[model_name] = spacy.load(path_or_func)
        else:
            self.name_to_model[model_name] = path_or_func()
        for call_back in self.name_to_reload_callbacks[model_name]:
            call_back()
