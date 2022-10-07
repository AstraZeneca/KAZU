from typing import Dict, Iterable, Union, List

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
    parser_name_to_entity_type: Dict[str, str],
    labels: Union[Iterable[str], str],
    output_dir: PathLike,
    parsers: List[OntologyParser] = None,
    curated_list: PathLike = None,
    span_key: str = SPAN_KEY,
) -> spacy.language.Language:
    """Generates, serializes and returns a Spacy pipeline with an :class:`~kazu.modelling.ontology_matching.ontology_matcher.OntologyMatcher`.

    Generates an English Spacy pipeline with a tokenizer, a sentencizer with default
    config, and an OntologyMatcher based on the input parameters. The pipeline is
    written to disk, and also returned to the caller.

    This is built with either a set of parsers in the case of a 'from scratch' build, for a new parser for example, or with a curated/annotated
    list of synonyms.

    :param parser_name_to_entity_type: a dictionary from parser names to the entity class we want
        to associate them with.
    :param labels: the entity class labels that the
        :class:`kazu.steps.joint_ner_and_linking.explosion.ExplosionStringMatchingStep` can
        recognise - this affects the matchers that are generated.
    :param output_dir: the output directory to write the pipeline into.
    :param parsers: parsers provided to build pipeline from using their associated synonyms.
    :param curated_list: curated list to build pipeline from using the synonyms in the list.
    :param span_key: the key to use within the generated Spacy Docs'
        `span attribute <https://spacy.io/api/doc#spans>`_ to store and access recognised NER
        spans.
    """

    if parsers is None and curated_list is None:
        raise ValueError(
            "Exactly one of parsers and curated_list needs to be provided (and not None) to get any synonyms to build the OntologyMatcher"
        )
    elif parsers is not None and curated_list is not None:
        raise ValueError(
            "Both parsers and curated_list passed - we build the pipeline from either one of these at a time, not both."
        )

    nlp = spacy.blank("en")
    custom_tokenizer(nlp)
    nlp.add_pipe("sentencizer")
    config = {"span_key": span_key, "parser_name_to_entity_type": parser_name_to_entity_type}
    ontology_matcher = nlp.add_pipe("ontology_matcher", config=config)
    assert isinstance(ontology_matcher, OntologyMatcher)
    if isinstance(labels, str):
        labels = labels.split(",")
    ontology_matcher.set_labels(labels)

    if parsers is not None:
        ontology_matcher.create_lowercase_phrasematcher_from_parsers(parsers)
    else:
        assert curated_list is not None
        ontology_matcher.create_phrasematchers_from_curated_list(curated_list)

    nlp.to_disk(output_dir)
    return nlp
