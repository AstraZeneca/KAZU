import typer
import spacy

from modelling.ontology_matching.ontology_matcher import OntologyMatcher
from kazu.utils.utils import PathLike, SinglePathLikeOrIterable


def main(
    parquet_files: SinglePathLikeOrIterable, labels: str, span_key: str, output_dir: PathLike
) -> spacy.language.Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    config = {"span_key": span_key}
    ontology_matcher = nlp.add_pipe("ontology_matcher", config=config)
    assert isinstance(ontology_matcher, OntologyMatcher)
    ontology_matcher.set_labels(labels.split(","))
    ontology_matcher.set_ontologies(parquet_files)
    nlp.to_disk(output_dir)
    return nlp


if __name__ == "__main__":
    typer.run(main)
