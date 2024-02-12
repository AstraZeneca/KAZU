import dataclasses
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
    STRINGMATCHING_EXAMPLE_TEXT,
    STRINGMATCHING_PARAM_NAMES,
    STRINGMATCHING_PARAM_VALUES,
    FIRST_MOCK_PARSER,
    SECOND_MOCK_PARSER,
    FIRST_MOCK_PARSER_DEFAULT_COMPLEX7_TERM,
    SECOND_MOCK_PARSER_DEFAULT_COMPLEX7_TERM,
    ENT_TYPE_1,
    COMPLEX_7_DISEASE_ALPHA_NORM,
    MatchOntologyData,
    StringMatchingTestCase,
    convert_test_case_to_param,
)
from kazu.tests.utils import DummyParser, write_curations, ignore_all_by_default_autocurator_factory
from kazu.utils.utils import Singleton

pytestmark = pytest.mark.usefixtures(
    "mock_kazu_disk_cache_on_parsers", "mock_build_fast_string_matcher_cache"
)


max_mention_test_case = StringMatchingTestCase(
    id="Both curations for same string and entity class Hit should get higher MentionConfidence",
    parser_1_curations=[
        dataclasses.replace(
            FIRST_MOCK_PARSER_DEFAULT_COMPLEX7_TERM,
            original_forms=frozenset(
                dataclasses.replace(form, mention_confidence=MentionConfidence.PROBABLE)
                for form in FIRST_MOCK_PARSER_DEFAULT_COMPLEX7_TERM.original_forms
            ),
        ),
    ],
    parser_2_curations=[
        dataclasses.replace(
            SECOND_MOCK_PARSER_DEFAULT_COMPLEX7_TERM,
            original_forms=frozenset(
                dataclasses.replace(
                    form,
                    mention_confidence=MentionConfidence.PROBABLE,
                    case_sensitive=True,
                    string="ComplexVII Disease\u03B1",
                )
                for form in FIRST_MOCK_PARSER_DEFAULT_COMPLEX7_TERM.original_forms
            ),
        )
    ],
    match_len=1,
    match_texts={"ComplexVII Disease\u03B1"},
    match_ontology_data={
        (
            ENT_TYPE_1,
            FIRST_MOCK_PARSER,
            COMPLEX_7_DISEASE_ALPHA_NORM,
            MentionConfidence.HIGHLY_LIKELY,
        ),
        (
            ENT_TYPE_1,
            SECOND_MOCK_PARSER,
            COMPLEX_7_DISEASE_ALPHA_NORM,
            MentionConfidence.HIGHLY_LIKELY,
        ),
    },
    # they need to be the same entity type
    # to get aggregated together
    parser_2_ent_type=ENT_TYPE_1,
)

mem_efficient_param_values = STRINGMATCHING_PARAM_VALUES + [
    convert_test_case_to_param(max_mention_test_case)
]


@pytest.mark.parametrize(STRINGMATCHING_PARAM_NAMES, mem_efficient_param_values)
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
        autocurator=ignore_all_by_default_autocurator_factory(),
    )
    parser_2 = DummyParser(
        name="second_mock_parser",
        entity_class=parser_2_ent_type,
        source=SECOND_MOCK_PARSER,
        curations_path=str(TEST_CURATIONS_PATH_PARSER_2),
        data=parser_2_data,
        autocurator=ignore_all_by_default_autocurator_factory(),
    )
    step = MemoryEfficientStringMatchingStep(parsers=[parser_1, parser_2])
    success, _failed = step([Document.create_simple_document(STRINGMATCHING_EXAMPLE_TEXT)])
    entities = success[0].get_entities()
    assert_matches(entities, match_len, match_texts, match_ontology_data)


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

    step = MemoryEfficientStringMatchingStep(parsers=[parser_1, parser_2, parser_3])
    success, _failed = step([Document.create_simple_document(STRINGMATCHING_EXAMPLE_TEXT)])
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
        ("ent_type_1", "first_mock_parser", "Q42_SYN", MentionConfidence.PROBABLE),
        ("ent_type_2", "second_mock_parser", "Q8_SYN", MentionConfidence.PROBABLE),
        ("ent_type_3", "third_mock_parser", "SYNONYMTERM", MentionConfidence.PROBABLE),
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
        for syn_term in e.syn_term_to_synonym_terms:
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
        assert e.match == STRINGMATCHING_EXAMPLE_TEXT[e.start : e.end]
    assert len(entities) == match_len
    assert set(e.match for e in entities) == match_texts
    converted_matches = convert_entities_to_match_ontology_data(entities)
    assert converted_matches == match_ontology_data
