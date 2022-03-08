import pytest
from pathlib import Path
import pandas as pd

import spacy
from spacy.lang.en import English
from spacy.tests.util import make_tempdir

from kazu.modelling.ontology_matching.ontology_matcher import OntologyMatcher, DISEASE, ANATOMY
from kazu.modelling.ontology_preprocessing.base import DEFAULT_LABEL, IDX, SYN


@pytest.fixture(scope="session")
def example_text():
    return "There is a Q42_ID and Q42_syn in this sentence, as well as Q42_syn & Q8_syn synonyms and label Q8_label."


def _create_parquet_file(file_loc):
    df = pd.DataFrame.from_dict(
        {
            IDX: [
                "http://purl.obolibrary.org/obo/UBERON_042",
                "http://purl.obolibrary.org/obo/MONDO_08",
            ],
            DEFAULT_LABEL: ["Q42_label", "Q8_label"],
            SYN: ["Q42_syn", "Q8_syn"],
        }
    )
    df.to_parquet(file_loc)
    return file_loc


def test_constructor():
    with make_tempdir() as d:
        nlp = English()
        default_config = {
            "span_key": "my_results",
            "entry_filter": {"@misc": "arizona.entry_filter_blacklist.v1"},
            "variant_generator": {"@misc": "arizona.variant_generator.v1"},
        }
        ontology_matcher = OntologyMatcher(nlp, d, **default_config)
        assert ontology_matcher.span_key == "my_results"
        assert ontology_matcher.nr_strict_rules == 0
        assert ontology_matcher.nr_lowercase_rules == 0
        assert ontology_matcher.labels == []
        assert ontology_matcher.ontologies == []


def test_initialize():
    with make_tempdir() as d:
        nlp = English()
        ontology_matcher = nlp.add_pipe("ontology_matcher")
        assert isinstance(ontology_matcher, OntologyMatcher)
        # no matcher rules are defined
        nlp.initialize()
        assert ontology_matcher.nr_strict_rules == 0
        assert ontology_matcher.nr_lowercase_rules == 0
        # non-existing dir does not work
        with pytest.raises(ValueError):
            ontology_matcher.set_ontologies(d / "non_existing")
        # non-existing file does not work
        with pytest.raises(ValueError):
            ontology_matcher.set_ontologies(d / "single_file.parquet")
        # existing parquet file
        parquet_loc = _create_parquet_file(d / "single_file.parquet")
        ontology_matcher.set_ontologies(parquet_loc)
        assert ontology_matcher.ontologies == ["single_file.parquet"]
        assert ontology_matcher.nr_strict_rules > 0
        assert ontology_matcher.nr_lowercase_rules > 0
        # single json file does not work
        json_loc = d / "single_file.json"
        Path(json_loc).touch()
        with pytest.raises(ValueError):
            ontology_matcher.set_ontologies(json_loc)
        # subdir of files: ignore non-parquet files
        sub_dir = d / "subdir"
        sub_dir.mkdir()
        _create_parquet_file(sub_dir / "file1.parquet")
        Path(sub_dir / "file2.json").touch()
        _create_parquet_file(sub_dir / "file3.parquet")
        ontology_matcher.set_ontologies(sub_dir)
        assert set(ontology_matcher.ontologies) == set(("file1.parquet", "file3.parquet"))


def test_apply(example_text):
    with make_tempdir() as d:
        nlp = English()
        parquet_loc = d / "onto_synonyms.parquet"
        parquet_file = _create_parquet_file(parquet_loc)
        config = {"span_key": "my_hits"}
        ontology_matcher = nlp.add_pipe("ontology_matcher", config=config)
        assert isinstance(ontology_matcher, OntologyMatcher)
        ontology_matcher.set_labels([ANATOMY, DISEASE])
        ontology_matcher.set_ontologies(parquet_file)
        doc = nlp(example_text)
        matches = doc.spans["my_hits"]
        assert len(matches) == 3
        assert set([m.text for m in matches]) == {"Q42_syn", "Q8_syn"}
        assert set([m.label_ for m in matches]) == {ANATOMY, DISEASE}
        assert set([m.kb_id_ for m in matches]) == {
            "http://purl.obolibrary.org/obo/UBERON_042",
            "http://purl.obolibrary.org/obo/MONDO_08",
        }


# fmt: off
@pytest.mark.parametrize(
    "labels,results",
    [
        ([], 0),
        ([ANATOMY], 2),
        ([DISEASE], 1),
        ([ANATOMY, DISEASE], 3),
        (["RANDOM LABEL", ANATOMY, DISEASE], 3),
    ]
)
# fmt: on
def test_labels(example_text, labels, results):
    with make_tempdir() as d:
        nlp = English()
        parquet_loc = d / "onto_synonyms.parquet"
        parquet_file = _create_parquet_file(parquet_loc)
        config = {"span_key": "my_hits"}
        ontology_matcher = nlp.add_pipe("ontology_matcher", config=config)
        assert isinstance(ontology_matcher, OntologyMatcher)
        ontology_matcher.set_labels(labels)
        ontology_matcher.set_ontologies(parquet_file)
        doc = nlp(example_text)
        matches = doc.spans["my_hits"]
        assert len(matches) == results


def test_serialization(example_text):
    with make_tempdir() as d:
        nlp1 = English()
        parquet_loc = d / "onto_synonyms.parquet"
        parquet_file = _create_parquet_file(parquet_loc)
        config = {"span_key": "my_hits"}
        ontology_matcher = nlp1.add_pipe("ontology_matcher", config=config)
        assert isinstance(ontology_matcher, OntologyMatcher)
        ontology_matcher.set_labels([ANATOMY, DISEASE])
        ontology_matcher.set_ontologies(parquet_file)
        doc1 = nlp1(example_text)
        spans1 = set((s.start, s.end, s.text) for s in doc1.spans["my_hits"])
        nlp_loc = d / "ontology_pipeline"
        nlp1.to_disk(nlp_loc)
        nlp2 = spacy.load(nlp_loc)
        doc2 = nlp2(example_text)
        spans2 = set((s.start, s.end, s.text) for s in doc2.spans["my_hits"])
        assert len(spans1) == 3
        assert len(spans2) == 3
        assert spans1 == spans2
