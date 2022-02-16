from typing import Iterable, Union

import typer
import spacy

from kazu.modelling.ontology_matching.ontology_matcher import OntologyMatcher, SPAN_KEY
from kazu.utils.utils import PathLike, SinglePathLikeOrIterable


def main(
    parquet_files: SinglePathLikeOrIterable,
    labels: Union[Iterable[str], str],
    output_dir: PathLike,
    span_key: str = SPAN_KEY,
) -> spacy.language.Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    config = {"span_key": span_key}
    ontology_matcher = nlp.add_pipe("ontology_matcher", config=config)
    assert isinstance(ontology_matcher, OntologyMatcher)
    if isinstance(labels, str):
        labels = labels.split(",")
    ontology_matcher.set_labels(labels)
    ontology_matcher.set_ontologies(parquet_files)
    nlp.to_disk(output_dir)
    return nlp


if __name__ == "__main__":
    typer.run(main)
