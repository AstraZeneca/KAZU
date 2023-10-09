import dataclasses

import pytest
from kazu.data.data import (
    CuratedTerm,
    MentionConfidence,
    CuratedTermBehaviour,
    EquivalentIdSet,
    Document,
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
from kazu.tests.utils import DummyParser, write_curations
from kazu.utils.utils import Singleton

pytestmark = pytest.mark.usefixtures(
    "mock_kazu_disk_cache_on_parsers", "mock_build_fast_string_matcher_cache"
)

example_text = """There is a Q42_ID and Q42_syn in this sentence, as well as Q42_syn & Q8_syn synonyms.
    This sentence is just to test when there are multiple synonyms for a single SynonymTerm,
    like for complex 7 disease alpha a.k.a ComplexVII Disease\u03B1 amongst others."""


FIRST_MOCK_PARSER = "first_mock_parser"
SECOND_MOCK_PARSER = "second_mock_parser"
COMPLEX_7_DISEASE_ALPHA_NORM = "COMPLEX 7 DISEASE ALPHA"
TARGET_IDX = "http://my.fake.ontology/complex_disease_123"
ENT_TYPE_1 = "ent_type_1"
ENT_TYPE_2 = "ent_type_2"
PARSER_1_DEFAULT_DATA = {
    IDX: [
        "http://my.fake.ontology/synonym_term_id_123",
        TARGET_IDX,
        TARGET_IDX,
        "http://my.fake.ontology_amongst_id_123",
        "http://my.fake.ontology_amongst_id_124",
    ],
    DEFAULT_LABEL: [
        "SynonymTerm",
        "SynonymTerm",
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
}

PARSER_2_DEFAULT_DATA = {
    IDX: [
        "http://my.fake.ontology/synonym_term_id_123",
        "http://my.fake.ontology/synonym_term_id_456",
        TARGET_IDX,
        "http://my.fake.ontology_amongst_id_123",
    ],
    DEFAULT_LABEL: [
        "SynonymTerm",
        "SynonymTerm",
        "Complex Disease Alpha VII",
        "Amongst",
    ],
    SYN: ["SynonymTerm", "SynonymTerm", "complexVII disease\u03B1", "amongst"],
    MAPPING_TYPE: ["test", "test", "test", "test"],
}

FIRST_MOCK_PARSER_DEFAULT_COMPLEX7_TERM = CuratedTerm(
    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
    curated_synonym="complexVII disease\u03B1",
    behaviour=CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
    associated_id_sets=frozenset(
        [
            EquivalentIdSet(
                ids_and_source=frozenset(
                    [
                        (
                            TARGET_IDX,
                            FIRST_MOCK_PARSER,
                        )
                    ]
                )
            )
        ]
    ),
    case_sensitive=False,
)

SECOND_MOCK_PARSER_DEFAULT_COMPLEX7_TERM = dataclasses.replace(
    FIRST_MOCK_PARSER_DEFAULT_COMPLEX7_TERM,
    associated_id_sets=frozenset(
        [
            EquivalentIdSet(
                ids_and_source=frozenset(
                    [
                        (
                            TARGET_IDX,
                            SECOND_MOCK_PARSER,
                        )
                    ]
                )
            )
        ]
    ),
)


@pytest.mark.parametrize(
    (
        "parser_1_curations",
        "parser_2_curations",
        "match_len",
        "match_texts",
        "match_ontology_dicts",
        "parser_1_data",
        "parser_2_data",
    ),
    [
        pytest.param(
            [FIRST_MOCK_PARSER_DEFAULT_COMPLEX7_TERM],
            [SECOND_MOCK_PARSER_DEFAULT_COMPLEX7_TERM],
            2,
            {"ComplexVII Disease\u03B1"},
            [
                {
                    ENT_TYPE_1: {
                        (
                            FIRST_MOCK_PARSER,
                            COMPLEX_7_DISEASE_ALPHA_NORM,
                            str(MentionConfidence.HIGHLY_LIKELY.value),
                        )
                    }
                },
                {
                    ENT_TYPE_2: {
                        (
                            SECOND_MOCK_PARSER,
                            COMPLEX_7_DISEASE_ALPHA_NORM,
                            str(MentionConfidence.HIGHLY_LIKELY.value),
                        )
                    }
                },
            ],
            PARSER_1_DEFAULT_DATA,
            PARSER_2_DEFAULT_DATA,
            id="Two curated case insensitive terms from two parsers Both should hit",
        ),
        pytest.param(
            [FIRST_MOCK_PARSER_DEFAULT_COMPLEX7_TERM],
            [dataclasses.replace(SECOND_MOCK_PARSER_DEFAULT_COMPLEX7_TERM, case_sensitive=True)],
            1,
            {"ComplexVII Disease\u03B1"},
            [
                {
                    ENT_TYPE_1: {
                        (
                            FIRST_MOCK_PARSER,
                            COMPLEX_7_DISEASE_ALPHA_NORM,
                            str(MentionConfidence.HIGHLY_LIKELY.value),
                        )
                    }
                },
            ],
            PARSER_1_DEFAULT_DATA,
            PARSER_2_DEFAULT_DATA,
            id="Two curated terms from two parsers One should hit to test case sensitivity",
        ),
        pytest.param(
            [FIRST_MOCK_PARSER_DEFAULT_COMPLEX7_TERM],
            [
                dataclasses.replace(
                    SECOND_MOCK_PARSER_DEFAULT_COMPLEX7_TERM, behaviour=CuratedTermBehaviour.IGNORE
                )
            ],
            1,
            {"ComplexVII Disease\u03B1"},
            [
                {
                    ENT_TYPE_1: {
                        (
                            FIRST_MOCK_PARSER,
                            COMPLEX_7_DISEASE_ALPHA_NORM,
                            str(MentionConfidence.HIGHLY_LIKELY.value),
                        )
                    }
                },
            ],
            PARSER_1_DEFAULT_DATA,
            PARSER_2_DEFAULT_DATA,
            id="Two curated terms from two parsers One should hit to test ignore logic",
        ),
        pytest.param(
            [
                dataclasses.replace(
                    FIRST_MOCK_PARSER_DEFAULT_COMPLEX7_TERM,
                    curated_synonym="This sentence is just to test",
                )
            ],
            [],
            1,
            {"This sentence is just to test"},
            [
                {
                    ENT_TYPE_1: {
                        (
                            FIRST_MOCK_PARSER,
                            "THIS SENTENCE IS JUST TO TEST",
                            str(MentionConfidence.HIGHLY_LIKELY.value),
                        )
                    }
                },
            ],
            PARSER_1_DEFAULT_DATA,
            PARSER_2_DEFAULT_DATA,
            id="One curated term with a novel synonym This should be added to the synonym DB and hit",
        ),
    ],
)
def test_pipeline_build_from_parsers_and_curated_list(
    tmp_path,
    parser_1_curations,
    parser_2_curations,
    match_len,
    match_texts,
    match_ontology_dicts,
    parser_1_data,
    parser_2_data,
):

    Singleton.clear_all()
    TEST_CURATIONS_PATH_PARSER_1 = tmp_path / "parser1_curated_terms.jsonl"
    TEST_CURATIONS_PATH_PARSER_2 = tmp_path / "parser2_curated_terms.jsonl"
    write_curations(path=TEST_CURATIONS_PATH_PARSER_1, terms=parser_1_curations)
    write_curations(path=TEST_CURATIONS_PATH_PARSER_2, terms=parser_2_curations)
    parser_1 = DummyParser(
        name=FIRST_MOCK_PARSER,
        entity_class="ent_type_1",
        source=FIRST_MOCK_PARSER,
        curations_path=str(TEST_CURATIONS_PATH_PARSER_1),
        data=parser_1_data,
    )
    parser_2 = DummyParser(
        name="second_mock_parser",
        entity_class="ent_type_2",
        source=SECOND_MOCK_PARSER,
        curations_path=str(TEST_CURATIONS_PATH_PARSER_2),
        data=parser_2_data,
    )
    step = MemoryEfficientStringMatchingStep(parsers=[parser_1, parser_2])
    success, failed = step([Document.create_simple_document(example_text)])
    entities = success[0].get_entities()
    assert_matches(entities, match_len, match_texts, match_ontology_dicts)


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
    success, failed = step([Document.create_simple_document(example_text)])
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
    match_ontology_dicts = [
        {"ent_type_1": {("first_mock_parser", "Q42_SYN", str(MentionConfidence.POSSIBLE.value))}},
        {"ent_type_1": {("first_mock_parser", "Q42_SYN", str(MentionConfidence.POSSIBLE.value))}},
        {"ent_type_2": {("second_mock_parser", "Q8_SYN", str(MentionConfidence.POSSIBLE.value))}},
        {
            "ent_type_3": {
                ("third_mock_parser", "SYNONYMTERM", str(MentionConfidence.POSSIBLE.value))
            }
        },
        {
            "ent_type_3": {
                (
                    "third_mock_parser",
                    "COMPLEX 7 DISEASE ALPHA",
                    str(MentionConfidence.PROBABLE.value),
                )
            }
        },
        {
            "ent_type_3": {
                (
                    "third_mock_parser",
                    "COMPLEX 7 DISEASE ALPHA",
                    str(MentionConfidence.PROBABLE.value),
                )
            }
        },
        {"ent_type_3": {("third_mock_parser", "AMONGST", str(MentionConfidence.PROBABLE.value))}},
    ]

    assert_matches(entities, match_len, match_texts, match_ontology_dicts)


def assert_matches(matches, match_len, match_texts, match_ontology_dicts):
    assert len(matches) == match_len
    assert set(m.match for m in matches) == match_texts
    match_data = set()
    parser_data = set()
    for m in matches:
        assert m.match == example_text[m.start : m.end + 1]

        syn_term = next(iter(m.syn_term_to_synonym_terms))
        match_data.add(
            (
                m.entity_class,
                syn_term.parser_name,
                syn_term.term_norm,
                str(m.mention_confidence.value),
            )
        )
    for item in match_ontology_dicts:
        for k, v in item.items():
            parser_data.add((k,) + next(iter(v)))

    assert match_data == parser_data
