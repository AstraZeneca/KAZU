from typing import List

import spacy
from kazu.modelling.ontology_matching.ontology_matcher import OntologyMatcher, SPAN_KEY
from kazu.modelling.ontology_preprocessing.base import OntologyParser
from kazu.utils.utils import PathLike


def custom_tokenizer(nlp):
    custom_infixes = [r"\(", "/"]
    infixes = custom_infixes + nlp.Defaults.infixes
    infix_re = spacy.util.compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer


def main(
    output_dir: PathLike,
    parsers: List[OntologyParser],
    curated_list: PathLike = None,
    span_key: str = SPAN_KEY,
) -> spacy.language.Language:
    """Generates, serializes and returns a Spacy pipeline with an
    :class:`~kazu.modelling.ontology_matching.ontology_matcher.OntologyMatcher`.

    Generates an English Spacy pipeline with a tokenizer, a sentencizer with default
    config, and an OntologyMatcher based on the input parameters. The pipeline is
    written to disk, and also returned to the caller.

    If 'parsers' is provided without an argument for 'curated_list', the
    :class:`~kazu.modelling.ontology_matching.ontology_matcher.OntologyMatcher` is built directly
    from the configured synonyms from the parsers (with any associated generated synonyms). This is
    useful for trying to understand which strings are 'noisy', but not recommended for production as raw ontology
    data tends to need some curation before it can be applied.

    If a curated_list is also provided, the build process will attempt to match these to the
    :class:`~kazu.data.data.SynonymTerm`\\ s coming from each parser. If a curated item can't be matched
    this will be logged as a warning.

    :param output_dir: the output directory to write the pipeline into.
    :param parsers: build the pipeline using these parsers as a data source.
    :param curated_list: optional path to a jsonl file. Each line should be json with fields:
        'term', 'case_sensitive', entity_class.
    :param span_key: the key to use within the generated Spacy Docs'
        `span attribute <https://spacy.io/api/doc#spans>`_ to store and access recognised NER
        spans.
    """
    nlp = spacy.blank("en")
    custom_tokenizer(nlp)
    nlp.add_pipe("sentencizer")
    config = {
        "span_key": span_key,
        "parser_name_to_entity_type": {parser.name: parser.entity_class for parser in parsers},
    }
    ontology_matcher = nlp.add_pipe("ontology_matcher", config=config)
    assert isinstance(ontology_matcher, OntologyMatcher)

    if curated_list is None:
        ontology_matcher.create_lowercase_phrasematcher_from_parsers(parsers)
    else:
        ontology_matcher.create_phrasematchers_from_curated_list(curated_list, parsers)

    nlp.to_disk(output_dir)
    return nlp
