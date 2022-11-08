import pytest
import spacy
from spacy.lang.en import English

from kazu.modelling.ontology_matching.assemble_pipeline import main as assemble_pipeline
from kazu.modelling.ontology_matching.ontology_matcher import OntologyMatcher
from kazu.modelling.ontology_preprocessing.base import IDX, DEFAULT_LABEL, SYN, MAPPING_TYPE
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


parser_1 = DummyParser(
    in_path="",
    name="first_mock_parser",
    source="test",
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
    in_path="",
    name="second_mock_parser",
    source="test",
    data={
        IDX: ["http://purl.obolibrary.org/obo/MONDO_08", "http://purl.obolibrary.org/obo/MONDO_08"],
        DEFAULT_LABEL: ["Q8", "Q8"],
        SYN: ["Q8_label", "Q8_syn"],
        MAPPING_TYPE: ["test", "test"],
    },
)

parser_3 = DummyParser(
    in_path="",
    name="third_mock_parser",
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

example_text = """There is a Q42_ID and Q42_syn in this sentence, as well as Q42_syn & Q8_syn synonyms.
    This sentence is just to test when there are multiple synonyms for a single SynonymTerm,
    like for complex 7 disease alpha a.k.a complexVII disease\u03B1 amongst others."""


@pytest.mark.parametrize(
    (
        "labels",
        "match_len",
        "match_texts",
        "match_ontology_dicts",
    ),
    [
        ([], 0, set(), []),
        (
            ["ent_type_1"],
            2,
            {"Q42_syn"},
            [
                {"ent_type_1": {("first_mock_parser", "Q42_SYN")}},
                {"ent_type_1": {("first_mock_parser", "Q42_SYN")}},
            ],
        ),
        (
            ["ent_type_2"],
            1,
            {"Q8_syn"},
            [{"ent_type_2": {("second_mock_parser", "Q8_SYN")}}],
        ),
        (
            ["ent_type_1", "ent_type_2"],
            3,
            {"Q42_syn", "Q8_syn"},
            [
                {"ent_type_1": {("first_mock_parser", "Q42_SYN")}},
                {"ent_type_1": {("first_mock_parser", "Q42_SYN")}},
                {"ent_type_2": {("second_mock_parser", "Q8_SYN")}},
            ],
        ),
        (
            ["RANDOM LABEL", "ent_type_1", "ent_type_2"],
            3,
            {"Q42_syn", "Q8_syn"},
            [
                {"ent_type_1": {("first_mock_parser", "Q42_SYN")}},
                {"ent_type_1": {("first_mock_parser", "Q42_SYN")}},
                {"ent_type_2": {("second_mock_parser", "Q8_SYN")}},
            ],
        ),
        (
            ["ent_type_3"],
            4,
            {"SynonymTerm", "complex 7 disease alpha", "complexVII disease\u03B1", "amongst"},
            [
                {"ent_type_3": {("third_mock_parser", "SYNONYMTERM")}},
                {"ent_type_3": {("third_mock_parser", "COMPLEX 7 DISEASE ALPHA")}},
                {"ent_type_3": {("third_mock_parser", "COMPLEX 7 DISEASE ALPHA")}},
                {"ent_type_3": {("third_mock_parser", "AMONGST")}},
            ],
        ),
    ],
)
def test_results_and_serialization(
    tmp_path,
    labels,
    match_len,
    match_texts,
    match_ontology_dicts,
):
    TEST_SPAN_KEY = "my_hits"
    TEST_OUTPUT_DIR = tmp_path / "ontology_pipeline"
    parser_name_to_entity_type = {
        parser_1.name: "ent_type_1",
        parser_2.name: "ent_type_2",
        parser_3.name: "ent_type_3",
    }
    nlp = assemble_pipeline(
        parsers=[parser_1, parser_2, parser_3],
        parser_name_to_entity_type=parser_name_to_entity_type,
        labels=labels,
        output_dir=TEST_OUTPUT_DIR,
        span_key=TEST_SPAN_KEY,
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


def assert_matches(matches, match_len, match_texts, match_ontology_dicts):
    assert len(matches) == match_len
    assert set(m.text for m in matches) == match_texts
    assert [m._.ontology_dict_ for m in matches] == match_ontology_dicts
