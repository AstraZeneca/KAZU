import typer
import spacy

from modelling.ontology_matching.ontology_matcher import OntologyMatcher


def main(parquet_dir: str, labels: str, span_key: str, output_dir: str) -> spacy.language.Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    config = {"span_key": span_key}
    ontology_matcher = nlp.add_pipe("ontology_matcher", config=config)
    assert isinstance(ontology_matcher, OntologyMatcher)
    ontology_matcher.set_labels(labels.split(","))
    ontology_matcher.set_ontologies(parquet_dir)
    nlp.to_disk(output_dir)
    return nlp


if __name__ == "__main__":
    typer.run(main)
