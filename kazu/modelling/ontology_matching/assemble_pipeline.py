from typing import Iterable, Union, List

import spacy
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
    labels: Union[Iterable[str], str],
    output_dir: PathLike,
    span_key: str = SPAN_KEY,
) -> spacy.language.Language:
    nlp = spacy.blank("en")
    custom_tokenizer(nlp)
    nlp.add_pipe("sentencizer")
    config = {"span_key": span_key}
    ontology_matcher = nlp.add_pipe("ontology_matcher", config=config)
    assert isinstance(ontology_matcher, OntologyMatcher)
    if isinstance(labels, str):
        labels = labels.split(",")
    ontology_matcher.set_labels(labels)
    ontology_matcher.set_ontologies(parsers)
    nlp.to_disk(output_dir)
    return nlp
