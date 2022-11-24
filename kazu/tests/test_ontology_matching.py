import dataclasses
import json
from pathlib import Path
from typing import List, Optional

import pytest
import spacy
from spacy.lang.en import English

from kazu.data.data import SynonymTerm, EquivalentIdAggregationStrategy
from kazu.modelling.ontology_matching.assemble_pipeline import main as assemble_pipeline
from kazu.modelling.ontology_matching.ontology_matcher import OntologyMatcher, CuratedTerm
from kazu.modelling.ontology_preprocessing.base import IDX, DEFAULT_LABEL, SYN, MAPPING_TYPE
from kazu.modelling.ontology_preprocessing.synonym_generation import (
    SynonymGenerator,
    CombinatorialSynonymGenerator,
)
from kazu.tests.utils import DummyParser


def test_constructor():
    nlp = English()
    default_config = {
        "span_key": "my_results",
        "parser_name_to_entity_type": {},
    }
    ontology_matcher = OntologyMatcher(nlp, **default_config)
    assert ontology_matcher.span_key == "my_results"
    assert ontology_matcher.nr_strict_rules == 0
    assert ontology_matcher.nr_lowercase_rules == 0
    assert ontology_matcher.labels == []


def test_initialize():
    nlp = English()
    config = {"parser_name_to_entity_type": {}}
    ontology_matcher = nlp.add_pipe("ontology_matcher", config=config)
    assert isinstance(ontology_matcher, OntologyMatcher)
    # no matcher rules are defined
    nlp.initialize()
    assert ontology_matcher.nr_strict_rules == 0
    assert ontology_matcher.nr_lowercase_rules == 0


example_text = """There is a Q42_ID and Q42_syn in this sentence, as well as Q42_syn & Q8_syn synonyms.
    This sentence is just to test when there are multiple synonyms for a single SynonymTerm,
    like for complex 7 disease alpha a.k.a complexVII disease\u03B1 amongst others."""


def write_curations(path: Path, terms: List[CuratedTerm]):
    with open(path, "w") as f:
        for term in terms:
            f.write(json.dumps(term.__dict__) + "\n")


class DummySynGenerator(SynonymGenerator):
    """
    test generator to check that a generated SynonymTerm with .terms that
    map to an exising value of .terms in another SynonymTerm is handled correctly
    """

    @classmethod
    def call(cls, synonym: SynonymTerm) -> Optional[SynonymTerm]:
        if "amongst us" in synonym.terms:
            return dataclasses.replace(
                synonym,
                terms=frozenset(["amongst", "amongst us"]),
                aggregated_by=EquivalentIdAggregationStrategy.CUSTOM,
            )
        else:
            return None


@pytest.mark.parametrize(
    ("curated_terms", "match_len", "match_texts", "match_ontology_dicts", "syn_generator"),
    [
        (
            [
                CuratedTerm(
                    term="SynonymTerm",
                    action="keep",
                    case_sensitive=True,
                    entity_class="ent_type_1",
                ),
                CuratedTerm(
                    term="SynonymTerm",
                    action="keep",
                    case_sensitive=True,
                    entity_class="ent_type_2",
                ),
            ],
            2,
            {"SynonymTerm"},
            [
                {"ent_type_1": {("first_mock_parser", "SYNONYMTERM")}},
                {"ent_type_2": {("second_mock_parser", "SYNONYMTERM")}},
            ],
            None,
        ),
        (
            [
                CuratedTerm(
                    term="complexVII Disease\u03B1",
                    action="keep",
                    case_sensitive=False,
                    entity_class="ent_type_1",
                ),
                CuratedTerm(
                    term="ComplexVII disease\u03B1",
                    action="keep",
                    case_sensitive=True,
                    entity_class="ent_type_2",
                ),
            ],
            1,
            {"complexVII diseaseα"},
            [
                {"ent_type_1": {("first_mock_parser", "COMPLEX 7 DISEASE ALPHA")}},
            ],
            None,
        ),
        (
            [
                CuratedTerm(
                    term="complexVII Disease\u03B1",
                    action="keep",
                    case_sensitive=False,
                    entity_class="ent_type_1",
                ),
                CuratedTerm(
                    term="ComplexVII disease\u03B1",
                    action="drop",
                    case_sensitive=True,
                    entity_class="ent_type_2",
                ),
            ],
            1,
            {"complexVII diseaseα"},
            [
                {"ent_type_1": {("first_mock_parser", "COMPLEX 7 DISEASE ALPHA")}},
            ],
            None,
        ),
        (
            [
                CuratedTerm(
                    term="amongst",
                    action="keep",
                    case_sensitive=False,
                    entity_class="ent_type_1",
                ),
            ],
            1,
            {"amongst"},
            [
                {"ent_type_1": {("first_mock_parser", "AMONGST")}},
            ],
            CombinatorialSynonymGenerator(
                [DummySynGenerator()]
            ),  # this generates a term that matches an existing ontology term, so we can test
            # that we're only matching to one term_norm (should be handled by syn generator)
        ),
    ],
)
def test_pipeline_build_from_parsers_and_curated_list(
    tmp_path, curated_terms, match_len, match_texts, match_ontology_dicts, syn_generator
):
    parser_1 = DummyParser(
        name="first_mock_parser",
        entity_class="ent_type_1",
        source="test",
        data={
            IDX: [
                "http://my.fake.ontology/synonym_term_id_123",
                "http://my.fake.ontology/complex_disease_123",
                "http://my.fake.ontology/complex_disease_123",
                "http://my.fake.ontology_amongst_id_123",
                "http://my.fake.ontology_amongst_id_124",
            ],
            DEFAULT_LABEL: [
                "SynonymTerm",
                "Complex Disease Alpha VII",
                "Complex Disease Alpha VII",
                "Amongst",
                "Amongst Us",
            ],
            SYN: [
                "SynonymTerm",
                "SynonymTerm",
                "complexVII disease\u03B1",
                "amongst",
                "amongst us",
            ],
            MAPPING_TYPE: ["test", "test", "test", "test", "test"],
        },
        synonym_generator=syn_generator,
    )
    parser_2 = DummyParser(
        name="second_mock_parser",
        entity_class="ent_type_2",
        source="test",
        data={
            IDX: [
                "http://my.fake.ontology/synonym_term_id_123",
                "http://my.fake.ontology/complex_disease_123",
                "http://my.fake.ontology/complex_disease_123",
                "http://my.fake.ontology_amongst_id_123",
            ],
            DEFAULT_LABEL: [
                "SynonymTerm",
                "Complex Disease Alpha VII",
                "Complex Disease Alpha VII",
                "Amongst",
            ],
            SYN: ["SynonymTerm", "SynonymTerm", "complexVII disease\u03B1", "amongst"],
            MAPPING_TYPE: ["test", "test", "test", "test"],
        },
        synonym_generator=syn_generator,
    )
    TEST_SPAN_KEY = "my_hits"
    TEST_CURATIONS_PATH = tmp_path / "curated_terms.jsonl"
    TEST_OUTPUT_DIR = tmp_path / "ontology_pipeline"
    write_curations(path=TEST_CURATIONS_PATH, terms=curated_terms)
    nlp = assemble_pipeline(
        parsers=[parser_1, parser_2],
        output_dir=TEST_OUTPUT_DIR,
        span_key=TEST_SPAN_KEY,
        curated_list=TEST_CURATIONS_PATH,
    )

    doc = nlp(example_text)
    matches = doc.spans[TEST_SPAN_KEY]
    assert_matches(matches, match_len, match_texts, match_ontology_dicts)
    nlp2 = spacy.load(TEST_OUTPUT_DIR)
    doc2 = nlp2(example_text)
    matches2 = doc2.spans[TEST_SPAN_KEY]
    assert_matches(matches2, match_len, match_texts, match_ontology_dicts)

    assert set((m.start_char, m.end_char, m.text) for m in matches2) == set(
        (m.start_char, m.end_char, m.text) for m in matches
    )


def test_pipeline_build_from_parsers_alone(tmp_path):

    parser_1 = DummyParser(
        name="first_mock_parser",
        source="test",
        entity_class="ent_type_1",
        data={
            IDX: [
                "http://purl.obolibrary.org/obo/UBERON_042",
                "http://purl.obolibrary.org/obo/UBERON_042",
            ],
            DEFAULT_LABEL: ["Q42", "Q42"],
            SYN: ["Q42_label", "Q42_syn"],
            MAPPING_TYPE: ["test", "test"],
        },
    )

    parser_2 = DummyParser(
        name="second_mock_parser",
        entity_class="ent_type_2",
        source="test",
        data={
            IDX: [
                "http://purl.obolibrary.org/obo/MONDO_08",
                "http://purl.obolibrary.org/obo/MONDO_08",
            ],
            DEFAULT_LABEL: ["Q8", "Q8"],
            SYN: ["Q8_label", "Q8_syn"],
            MAPPING_TYPE: ["test", "test"],
        },
    )

    parser_3 = DummyParser(
        name="third_mock_parser",
        entity_class="ent_type_3",
        source="test",
        data={
            IDX: [
                "http://my.fake.ontology/synonym_term_id_123",
                "http://my.fake.ontology/complex_disease_123",
                "http://my.fake.ontology/complex_disease_123",
                "http://my.fake.ontology_amongst_id_123",
            ],
            DEFAULT_LABEL: [
                "SynonymTerm",
                "Complex Disease Alpha VII",
                "Complex Disease Alpha VII",
                "Amongst",
            ],
            SYN: ["SynonymTerm", "complex 7 disease alpha", "complexVII disease\u03B1", "amongst"],
            MAPPING_TYPE: ["test", "test", "test", "test"],
        },
    )

    TEST_SPAN_KEY = "my_hits"
    TEST_OUTPUT_DIR = tmp_path / "ontology_pipeline"
    nlp = assemble_pipeline(
        parsers=[parser_1, parser_2, parser_3],
        output_dir=TEST_OUTPUT_DIR,
        span_key=TEST_SPAN_KEY,
    )

    doc = nlp(example_text)
    matches = doc.spans[TEST_SPAN_KEY]

    match_len = 7
    match_texts = {
        "Q42_syn",
        "Q8_syn",
        "SynonymTerm",
        "complex 7 disease alpha",
        "complexVII disease\u03B1",
        "amongst",
    }
    match_ontology_dicts = [
        {"ent_type_1": {("first_mock_parser", "Q42_SYN")}},
        {"ent_type_1": {("first_mock_parser", "Q42_SYN")}},
        {"ent_type_2": {("second_mock_parser", "Q8_SYN")}},
        {"ent_type_3": {("third_mock_parser", "SYNONYMTERM")}},
        {"ent_type_3": {("third_mock_parser", "COMPLEX 7 DISEASE ALPHA")}},
        {"ent_type_3": {("third_mock_parser", "COMPLEX 7 DISEASE ALPHA")}},
        {"ent_type_3": {("third_mock_parser", "AMONGST")}},
    ]

    assert_matches(matches, match_len, match_texts, match_ontology_dicts)

    nlp2 = spacy.load(TEST_OUTPUT_DIR)

    doc2 = nlp2(example_text)
    matches2 = doc2.spans[TEST_SPAN_KEY]

    assert_matches(matches2, match_len, match_texts, match_ontology_dicts)

    assert set((m.start_char, m.end_char, m.text) for m in matches2) == set(
        (m.start_char, m.end_char, m.text) for m in matches
    )


def assert_matches(matches, match_len, match_texts, match_ontology_dicts):
    assert len(matches) == match_len
    assert set(m.text for m in matches) == match_texts
    assert [m._.ontology_dict_ for m in matches] == match_ontology_dicts
