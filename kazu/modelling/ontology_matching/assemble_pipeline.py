from typing import Dict, Iterable, Union, List

import spacy
from kazu.modelling.ontology_matching.blacklist.synonym_blacklisting import BlackLister
from kazu.modelling.ontology_matching.ontology_matcher import OntologyMatcher, SPAN_KEY
from kazu.modelling.ontology_preprocessing.base import OntologyParser
from kazu.utils.utils import PathLike
from spacy.util import compile_infix_regex


def custom_tokenizer(nlp):
    custom_infixes = [r"\(", "/"]
    infixes = custom_infixes + nlp.Defaults.infixes
    infix_re = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer


def main(
    parsers: List[OntologyParser],
    blacklisters: Dict[str, BlackLister],
    parser_name_to_entity_type: Dict[str, str],
    labels: Union[Iterable[str], str],
    output_dir: PathLike,
    span_key: str = SPAN_KEY,
) -> spacy.language.Language:
    """Generates, serializes and returns a Spacy pipeline with an :class:`~kazu.modelling.ontology_matching.ontology_matcher.OntologyMatcher`.

    Generates an English Spacy pipeline with a tokenizer, a sentencizer with default
    config, and an OntologyMatcher based on the input parameters. The pipeline is
    written to disk, and also returned to the caller.

    :param parsers: parsers provided to build pipeline from using their associated synonyms.
    :param blacklisters: the blacklister that should be used for a given parser (key is parser name).
        Any parsers not specified will not have any synonyms blacklisted via this method.
    :param parser_name_to_entity_type: a dictionary from parser names to the entity class we want to associate them with.
    :param labels: the entity class labels that the :class:`kazu.steps.ner.explosion.ExplosionNERStep` can recognise - this affects the matchers that are generated.
    :param output_dir: the output directory to write the pipeline into.
    :param span_key: the key to use within the generated Spacy Docs' `span attribute <https://spacy.io/api/doc#spans>`_ to store and access recognised NER spans.
    """
    nlp = spacy.blank("en")
    custom_tokenizer(nlp)
    nlp.add_pipe("sentencizer")
    config = {"span_key": span_key, "parser_name_to_entity_type": parser_name_to_entity_type}
    ontology_matcher = nlp.add_pipe("ontology_matcher", config=config)
    assert isinstance(ontology_matcher, OntologyMatcher)
    if isinstance(labels, str):
        labels = labels.split(",")
    ontology_matcher.set_labels(labels)
    ontology_matcher.create_phrasematchers(parsers, blacklisters)
    nlp.to_disk(output_dir)
    return nlp
