import spacy

from kazu.ontology_matching.ontology_matcher import OntologyMatcher, SPAN_KEY
from kazu.ontology_preprocessing.base import OntologyParser
from kazu.utils.utils import PathLike
from kazu.utils.spacy_pipeline import basic_spacy_pipeline


def main(
    output_dir: PathLike, parsers: list[OntologyParser], span_key: str = SPAN_KEY
) -> spacy.language.Language:
    """Generates, serializes and returns a spaCy pipeline with an
    :class:`~kazu.ontology_matching.ontology_matcher.OntologyMatcher`.

    Generates an English spaCy pipeline with a tokenizer, a sentencizer with default
    config, and an OntologyMatcher based on the input parameters. The pipeline is
    written to disk, and also returned to the caller.

    If a parser has no curations configured, the
    :class:`~kazu.ontology_matching.ontology_matcher.OntologyMatcher` is built directly
    from the configured synonyms from the parsers (with any associated generated synonyms). This is
    useful for trying to understand which strings are 'noisy', but not recommended for production as raw ontology
    data tends to need some curation before it can be applied.

    If a parser has curations configured, it will be built using the configured instances of :class:`~kazu.data.data.OntologyStringResource`
    associated with the parser. This is generally the recommended behaviour in production, although obviously
    requires a set of high quality curations to be effective

    :param output_dir: the output directory to write the pipeline into.
    :param parsers: build the pipeline using these parsers as a data source.
    :param span_key: the key to use within the generated spaCy Docs'
        `span attribute <https://spacy.io/api/doc#spans>`_ to store and access recognised NER
        spans.
    :return:
    """
    nlp = basic_spacy_pipeline()
    config = {
        "span_key": span_key,
        "parser_name_to_entity_type": {parser.name: parser.entity_class for parser in parsers},
    }
    ontology_matcher = nlp.add_pipe("ontology_matcher", config=config)
    assert isinstance(ontology_matcher, OntologyMatcher)
    ontology_matcher.create_phrasematchers(parsers)
    nlp.to_disk(output_dir)
    return nlp
