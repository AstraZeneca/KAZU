from collections.abc import Iterable

import pytest

from kazu.data.data import (
    MentionConfidence,
    Document,
    Entity,
)
from kazu.ontology_preprocessing.base import (
    IDX,
    DEFAULT_LABEL,
    SYN,
    MAPPING_TYPE,
)
from kazu.ontology_preprocessing.synonym_generation import CombinatorialSynonymGenerator
from kazu.steps.joint_ner_and_linking.memory_efficient_string_matching import (
    MemoryEfficientStringMatchingStep,
)
from kazu.tests.string_matching_utils import (
    PARAM_NAMES,
    PARAM_VALUES,
    FIRST_MOCK_PARSER,
    SECOND_MOCK_PARSER,
    MatchOntologyData,
)
from kazu.tests.utils import DummyParser, write_curations
from kazu.utils.utils import Singleton

pytestmark = pytest.mark.usefixtures(
    "mock_kazu_disk_cache_on_parsers", "mock_build_fast_string_matcher_cache"
)

example_text = """There is a Q42_ID and Q42_syn in this sentence, as well as Q42_syn & Q8_syn synonyms.
    This sentence is just to test when there are multiple synonyms for a single SynonymTerm,
    like for complex 7 disease alpha a.k.a ComplexVII Disease\u03B1 amongst others."""


@pytest.mark.parametrize(PARAM_NAMES, PARAM_VALUES)
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
    step = MemoryEfficientStringMatchingStep(parsers=[parser_1, parser_2])
    success, _failed = step([Document.create_simple_document(example_text)])
    entities = success[0].get_entities()
    assert_matches(entities, match_len, match_texts, match_ontology_data)


def test_pipeline_build_from_parsers_alone():
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

    step = MemoryEfficientStringMatchingStep(parsers=[parser_1, parser_2, parser_3])
    success, _failed = step([Document.create_simple_document(example_text)])
    entities = success[0].get_entities()

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
        (
            "ent_type_3",
            "third_mock_parser",
            "COMPLEX 7 DISEASE ALPHA",
            MentionConfidence.PROBABLE,
        ),
        ("ent_type_3", "third_mock_parser", "AMONGST", MentionConfidence.PROBABLE),
    }

    assert_matches(entities, match_len, match_texts, match_ontology_data)


def convert_entities_to_match_ontology_data(
    entities: Iterable[Entity],
) -> MatchOntologyData:
    res = set()
    for e in entities:
        syn_term = next(iter(e.syn_term_to_synonym_terms))
        res.add(
            (
                e.entity_class,
                syn_term.parser_name,
                syn_term.term_norm,
                e.mention_confidence,
            )
        )
    return res


def assert_matches(
    entities: list[Entity],
    match_len: int,
    match_texts: set[str],
    match_ontology_data: MatchOntologyData,
) -> None:
    for e in entities:
        assert e.match == example_text[e.start : e.end]
    assert len(entities) == match_len
    assert set(e.match for e in entities) == match_texts
    converted_matches = convert_entities_to_match_ontology_data(entities)
    assert converted_matches == match_ontology_data
