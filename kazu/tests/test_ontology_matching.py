from typing import Any

import pytest
import spacy

from kazu.data.data import (
    MentionConfidence,
)
from kazu.ontology_matching.assemble_pipeline import main as assemble_pipeline
from kazu.ontology_matching.ontology_matcher import OntologyMatcher
from kazu.ontology_preprocessing.base import (
    IDX,
    DEFAULT_LABEL,
    SYN,
    MAPPING_TYPE,
)
from kazu.ontology_preprocessing.synonym_generation import CombinatorialSynonymGenerator
from kazu.tests.string_matching_utils import (
    STRINGMATCHING_EXAMPLE_TEXT,
    STRINGMATCHING_PARAM_NAMES,
    STRINGMATCHING_PARAM_VALUES,
    FIRST_MOCK_PARSER,
    SECOND_MOCK_PARSER,
    MatchOntologyData,
)
from kazu.tests.utils import DummyParser, write_curations
from kazu.utils.spacy_pipeline import SPACY_DEFAULT_INFIXES
from kazu.utils.utils import Singleton
from spacy.lang.en import English
from spacy.lang.en.punctuation import TOKENIZER_INFIXES

pytestmark = pytest.mark.usefixtures("mock_kazu_disk_cache_on_parsers")


def test_constructor():
    nlp = English()
    ontology_matcher = OntologyMatcher(nlp, span_key="my_results", parser_name_to_entity_type={})
    assert ontology_matcher.span_key == "my_results"
    assert ontology_matcher.nr_strict_rules == 0
    assert ontology_matcher.nr_lowercase_rules == 0
    assert ontology_matcher.labels == []


def test_initialize():
    nlp = English()
    config: dict[str, Any] = {"parser_name_to_entity_type": {}}
    ontology_matcher = nlp.add_pipe("ontology_matcher", config=config)
    assert isinstance(ontology_matcher, OntologyMatcher)
    # no matcher rules are defined
    nlp.initialize()
    assert ontology_matcher.nr_strict_rules == 0
    assert ontology_matcher.nr_lowercase_rules == 0


@pytest.mark.parametrize(STRINGMATCHING_PARAM_NAMES, STRINGMATCHING_PARAM_VALUES)
def test_pipeline_build_from_parsers_and_curated_list(
    tmp_path,
    parser_1_curations,
    parser_2_curations,
    match_len,
    match_texts,
    match_ontology_data,
    parser_1_data,
    parser_2_data,
    parser_1_ent_type,
    parser_2_ent_type,
):

    Singleton.clear_all()
    TEST_CURATIONS_PATH_PARSER_1 = tmp_path / "parser1_curated_terms.jsonl"
    TEST_CURATIONS_PATH_PARSER_2 = tmp_path / "parser2_curated_terms.jsonl"
    write_curations(path=TEST_CURATIONS_PATH_PARSER_1, terms=parser_1_curations)
    write_curations(path=TEST_CURATIONS_PATH_PARSER_2, terms=parser_2_curations)
    parser_1 = DummyParser(
        name=FIRST_MOCK_PARSER,
        entity_class=parser_1_ent_type,
        source=FIRST_MOCK_PARSER,
        curations_path=str(TEST_CURATIONS_PATH_PARSER_1),
        data=parser_1_data,
    )
    parser_2 = DummyParser(
        name="second_mock_parser",
        entity_class=parser_2_ent_type,
        source=SECOND_MOCK_PARSER,
        curations_path=str(TEST_CURATIONS_PATH_PARSER_2),
        data=parser_2_data,
    )
    TEST_SPAN_KEY = "my_hits"
    TEST_OUTPUT_DIR = tmp_path / "ontology_pipeline"

    nlp = assemble_pipeline(
        parsers=[parser_1, parser_2],
        output_dir=TEST_OUTPUT_DIR,
        span_key=TEST_SPAN_KEY,
    )

    doc = nlp(STRINGMATCHING_EXAMPLE_TEXT)
    matches = doc.spans[TEST_SPAN_KEY]
    assert_matches(matches, match_len, match_texts, match_ontology_data)
    nlp2 = spacy.load(TEST_OUTPUT_DIR)
    doc2 = nlp2(STRINGMATCHING_EXAMPLE_TEXT)
    matches2 = doc2.spans[TEST_SPAN_KEY]
    assert_matches(matches2, match_len, match_texts, match_ontology_data)

    assert set((m.start_char, m.end_char, m.text) for m in matches2) == set(
        (m.start_char, m.end_char, m.text) for m in matches
    )


def test_pipeline_build_from_parsers_alone(tmp_path):
    Singleton.clear_all()
    parser_1 = DummyParser(
        name="first_mock_parser",
        source="test",
        synonym_generator=None,
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
        synonym_generator=CombinatorialSynonymGenerator([]),
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

    doc = nlp(STRINGMATCHING_EXAMPLE_TEXT)
    matches = doc.spans[TEST_SPAN_KEY]

    match_len = 7
    match_texts = {
        "Q42_syn",
        "Q8_syn",
        "SynonymTerm",
        "complex 7 disease alpha",
        "ComplexVII Disease\u03B1",
        "amongst",
    }
    match_ontology_data = {
        ("ent_type_1", "first_mock_parser", "Q42_SYN", MentionConfidence.POSSIBLE),
        ("ent_type_2", "second_mock_parser", "Q8_SYN", MentionConfidence.POSSIBLE),
        ("ent_type_3", "third_mock_parser", "SYNONYMTERM", MentionConfidence.POSSIBLE),
        ("ent_type_3", "third_mock_parser", "COMPLEX 7 DISEASE ALPHA", MentionConfidence.PROBABLE),
        ("ent_type_3", "third_mock_parser", "AMONGST", MentionConfidence.PROBABLE),
    }

    assert_matches(matches, match_len, match_texts, match_ontology_data)

    nlp2 = spacy.load(TEST_OUTPUT_DIR)

    doc2 = nlp2(STRINGMATCHING_EXAMPLE_TEXT)
    matches2 = doc2.spans[TEST_SPAN_KEY]

    assert_matches(matches2, match_len, match_texts, match_ontology_data)

    assert set((m.start_char, m.end_char, m.text) for m in matches2) == set(
        (m.start_char, m.end_char, m.text) for m in matches
    )


def convert_spacy_spans_to_match_ontology_data(
    spans: spacy.tokens.SpanGroup,
) -> MatchOntologyData:
    return set(
        (ent_class, hit[0], hit[1], MentionConfidence(int(hit[2])))
        for m in spans
        for ent_class, item in m._.ontology_dict_.items()
        for hit in item
    )


def assert_matches(
    spans: spacy.tokens.SpanGroup,
    match_len: int,
    match_texts: set[str],
    match_ontology_data: MatchOntologyData,
) -> None:
    assert len(spans) == match_len
    assert set(s.text for s in spans) == match_texts
    converted_matches = convert_spacy_spans_to_match_ontology_data(spans)
    assert converted_matches == match_ontology_data


def test_no_spacy_tokenization_update():
    # have this as a test rather than an assert because we don't want this to break things for our users -
    # it doesn't really matter to them if the tokenization is a bit outdated relative to spacy, we just
    # want to know as developers that we should look at updating.
    assert set(TOKENIZER_INFIXES) == set(
        SPACY_DEFAULT_INFIXES
    ), "Our tokenization rules are outdated to spacy's"
