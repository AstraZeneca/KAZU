from copy import deepcopy
from string import ascii_lowercase
from typing import List

import spacy
from spacy.lang.en import English, EnglishDefaults
from spacy.lang.en.tokenizer_exceptions import TOKENIZER_EXCEPTIONS
from spacy.lang.char_classes import (
    LIST_ELLIPSES,
    LIST_ICONS,
    CONCAT_QUOTES,
    ALPHA_LOWER,
    ALPHA_UPPER,
    ALPHA,
    HYPHENS,
)

from kazu.ontology_matching.ontology_matcher import OntologyMatcher, SPAN_KEY
from kazu.ontology_preprocessing.base import OntologyParser
from kazu.utils.utils import PathLike


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
    suffixes = EnglishDefaults.suffixes + [
        r"(?<=\b[{au}])\.".format(au=ALPHA_UPPER)
    ]  # type:ignore[operator] # because mypy doesn't know that EnglishDefaults.suffixes is
    # always a List[str]


@spacy.registry.languages("kazu_custom_en")
class KazuCustomEnglish(English):
    lang = "kazu_custom_en"
    Defaults = KazuCustomEnglishDefaults


def main(
    output_dir: PathLike, parsers: List[OntologyParser], span_key: str = SPAN_KEY
) -> spacy.language.Language:
    """Generates, serializes and returns a Spacy pipeline with an
    :class:`~kazu.ontology_matching.ontology_matcher.OntologyMatcher`.

    Generates an English Spacy pipeline with a tokenizer, a sentencizer with default
    config, and an OntologyMatcher based on the input parameters. The pipeline is
    written to disk, and also returned to the caller.

    If a parser has no curations configured, the
    :class:`~kazu.ontology_matching.ontology_matcher.OntologyMatcher` is built directly
    from the configured synonyms from the parsers (with any associated generated synonyms). This is
    useful for trying to understand which strings are 'noisy', but not recommended for production as raw ontology
    data tends to need some curation before it can be applied.

    If a parser has curations configured, it will be built using the configured instances of :class:`~kazu.data.data.CuratedTerm`
    associated with the parser. This is generally the recommended behaviour in production, although obviously
    requires a set of high quality curations to be effective

    :param output_dir: the output directory to write the pipeline into.
    :param parsers: build the pipeline using these parsers as a data source.
    :param span_key: the key to use within the generated Spacy Docs'
        `span attribute <https://spacy.io/api/doc#spans>`_ to store and access recognised NER
        spans.
    """
    nlp = spacy.blank("kazu_custom_en")

    nlp.add_pipe("sentencizer")
    config = {
        "span_key": span_key,
        "parser_name_to_entity_type": {parser.name: parser.entity_class for parser in parsers},
    }
    ontology_matcher = nlp.add_pipe("ontology_matcher", config=config)
    assert isinstance(ontology_matcher, OntologyMatcher)
    ontology_matcher.create_phrasematchers_using_curations(parsers)
    nlp.to_disk(output_dir)
    return nlp
