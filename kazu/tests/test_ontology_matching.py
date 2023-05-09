import json
from pathlib import Path
from typing import List, Dict, Any

import pytest
import spacy
from spacy.lang.en import English
from spacy.lang.en.punctuation import TOKENIZER_INFIXES

from kazu.data.data import (
    Curation,
    DocumentJsonUtils,
    MentionConfidence,
    SynonymTermAction,
    SynonymTermBehaviour,
    EquivalentIdSet,
)
from kazu.modelling.ontology_matching.assemble_pipeline import (
    main as assemble_pipeline,
    SPACY_DEFAULT_INFIXES,
)
from kazu.modelling.ontology_matching.ontology_matcher import OntologyMatcher
from kazu.modelling.ontology_preprocessing.base import (
    IDX,
    DEFAULT_LABEL,
    SYN,
    MAPPING_TYPE,
    CurationException,
    load_curated_terms,
    kazu_disk_cache,
)
from kazu.modelling.ontology_preprocessing.synonym_generation import CombinatorialSynonymGenerator
from kazu.tests.utils import DummyParser
from kazu.utils.utils import Singleton


def test_constructor():
    nlp = English()
    ontology_matcher = OntologyMatcher(nlp, span_key="my_results", parser_name_to_entity_type={})
    assert ontology_matcher.span_key == "my_results"
    assert ontology_matcher.nr_strict_rules == 0
    assert ontology_matcher.nr_lowercase_rules == 0
    assert ontology_matcher.labels == []


def test_initialize():
    nlp = English()
    config: Dict[str, Any] = {"parser_name_to_entity_type": {}}
    ontology_matcher = nlp.add_pipe("ontology_matcher", config=config)
    assert isinstance(ontology_matcher, OntologyMatcher)
    # no matcher rules are defined
    nlp.initialize()
    assert ontology_matcher.nr_strict_rules == 0
    assert ontology_matcher.nr_lowercase_rules == 0


example_text = """There is a Q42_ID and Q42_syn in this sentence, as well as Q42_syn & Q8_syn synonyms.
    This sentence is just to test when there are multiple synonyms for a single SynonymTerm,
    like for complex 7 disease alpha a.k.a ComplexVII Disease\u03B1 amongst others."""


def write_curations(path: Path, terms: List[Curation]):
    with open(path, "w") as f:
        for curation in terms:
            f.write(json.dumps(DocumentJsonUtils.obj_to_dict_repr(curation)) + "\n")


FIRST_MOCK_PARSER = "first_mock_parser"
SECOND_MOCK_PARSER = "second_mock_parser"
COMPLEX_7_DISEASE_ALPHA_NORM = "COMPLEX 7 DISEASE ALPHA"
TARGET_IDX = "http://my.fake.ontology/complex_disease_123"
CONFUSING_IDX = "http://my.fake.ontology/i_m_very_confused_123"
CONFUSING_SYNONYM = "IM_A_SYMBOL_TO_CONFUSE_THE_pARSERL0L0L0"
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

PARSER_1_AMBIGUOUS_DATA = {
    IDX: [
        CONFUSING_IDX,
        TARGET_IDX,
        TARGET_IDX,
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
        CONFUSING_SYNONYM,
        CONFUSING_SYNONYM,
        "complexVII disease\u03B1",
        "amongst",
        "amongst us",
    ],
    MAPPING_TYPE: ["test", "test", "test", "test", "test"],
}


@pytest.mark.parametrize(
    (
        "parser_1_curations",
        "parser_2_curations",
        "match_len",
        "match_texts",
        "match_ontology_dicts",
        "parser_1_data",
        "parser_2_data",
        "throws_curation_exception",
    ),
    [
        pytest.param(
            [
                Curation(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    curated_synonym="complexVII disease\u03B1",
                    actions=tuple(
                        [
                            SynonymTermAction(
                                behaviour=SynonymTermBehaviour.ADD_FOR_NER_AND_LINKING,
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
                            ),
                        ]
                    ),
                    case_sensitive=False,
                ),
            ],
            [
                Curation(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    curated_synonym="complexVII disease\u03B1",
                    actions=tuple(
                        [
                            SynonymTermAction(
                                behaviour=SynonymTermBehaviour.ADD_FOR_NER_AND_LINKING,
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
                            ),
                        ]
                    ),
                    case_sensitive=False,
                ),
            ],
            2,
            {"ComplexVII Disease\u03B1"},
            [
                {ENT_TYPE_1: {(FIRST_MOCK_PARSER, COMPLEX_7_DISEASE_ALPHA_NORM)}},
                {ENT_TYPE_2: {(SECOND_MOCK_PARSER, COMPLEX_7_DISEASE_ALPHA_NORM)}},
            ],
            PARSER_1_DEFAULT_DATA,
            PARSER_2_DEFAULT_DATA,
            False,
            id="Two curated case insensitive terms from two parsers Both should hit",
        ),
        pytest.param(
            [
                Curation(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    curated_synonym="complexVII disease\u03B1",
                    actions=tuple(
                        [
                            SynonymTermAction(
                                behaviour=SynonymTermBehaviour.ADD_FOR_NER_AND_LINKING,
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
                            ),
                        ]
                    ),
                    case_sensitive=False,
                )
            ],
            [
                Curation(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    curated_synonym="complexVII disease\u03B1",
                    actions=tuple(
                        [
                            SynonymTermAction(
                                behaviour=SynonymTermBehaviour.ADD_FOR_NER_AND_LINKING,
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
                            ),
                        ]
                    ),
                    case_sensitive=True,
                )
            ],
            1,
            {"ComplexVII Disease\u03B1"},
            [
                {ENT_TYPE_1: {(FIRST_MOCK_PARSER, COMPLEX_7_DISEASE_ALPHA_NORM)}},
            ],
            PARSER_1_DEFAULT_DATA,
            PARSER_2_DEFAULT_DATA,
            False,
            id="Two curated terms from two parsers One should hit to test case sensitivity",
        ),
        pytest.param(
            [
                Curation(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    curated_synonym="complexVII disease\u03B1",
                    actions=tuple(
                        [
                            SynonymTermAction(
                                behaviour=SynonymTermBehaviour.ADD_FOR_NER_AND_LINKING,
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
                            )
                        ]
                    ),
                    case_sensitive=False,
                ),
            ],
            [
                Curation(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    curated_synonym="others",
                    actions=tuple(
                        [
                            SynonymTermAction(
                                behaviour=SynonymTermBehaviour.ADD_FOR_NER_AND_LINKING,
                                associated_id_sets=frozenset(
                                    [
                                        EquivalentIdSet(
                                            ids_and_source=frozenset(
                                                [
                                                    (
                                                        "I dont exist",
                                                        SECOND_MOCK_PARSER,
                                                    )
                                                ]
                                            )
                                        )
                                    ]
                                ),
                            )
                        ]
                    ),
                    case_sensitive=False,
                ),
            ],
            0,
            {},
            [],
            PARSER_1_DEFAULT_DATA,
            PARSER_2_DEFAULT_DATA,
            True,
            id="Two curated terms from two parsers An exception should be thrown at build time as the second ID doesnt exist",
        ),
        pytest.param(
            [
                Curation(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    curated_synonym="complexVII disease\u03B1",
                    actions=tuple(
                        [
                            SynonymTermAction(
                                behaviour=SynonymTermBehaviour.ADD_FOR_NER_AND_LINKING,
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
                            )
                        ]
                    ),
                    case_sensitive=False,
                ),
            ],
            [
                Curation(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    curated_synonym="complexVII disease\u03B1",
                    actions=tuple(
                        [
                            SynonymTermAction(
                                behaviour=SynonymTermBehaviour.IGNORE,
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
                        ]
                    ),
                    case_sensitive=False,
                ),
            ],
            1,
            {"ComplexVII Disease\u03B1"},
            [
                {ENT_TYPE_1: {(FIRST_MOCK_PARSER, COMPLEX_7_DISEASE_ALPHA_NORM)}},
            ],
            PARSER_1_DEFAULT_DATA,
            PARSER_2_DEFAULT_DATA,
            False,
            id="Two curated terms from two parsers One should hit to test ignore logic",
        ),
        pytest.param(
            [
                Curation(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    curated_synonym="This sentence is just to test",
                    actions=tuple(
                        [
                            SynonymTermAction(
                                behaviour=SynonymTermBehaviour.ADD_FOR_NER_AND_LINKING,
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
                        ]
                    ),
                    case_sensitive=False,
                ),
            ],
            [],
            1,
            {"This sentence is just to test"},
            [
                {ENT_TYPE_1: {(FIRST_MOCK_PARSER, "THIS SENTENCE IS JUST TO TEST")}},
            ],
            PARSER_1_DEFAULT_DATA,
            PARSER_2_DEFAULT_DATA,
            False,
            id="One curated term with a novel synonym This should be added to the synonym DB and hit",
        ),
        pytest.param(
            [
                Curation(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    curated_synonym=CONFUSING_SYNONYM,
                    actions=tuple(
                        [
                            SynonymTermAction(
                                behaviour=SynonymTermBehaviour.ADD_FOR_NER_AND_LINKING,
                                associated_id_sets=frozenset(
                                    [
                                        EquivalentIdSet(
                                            ids_and_source=frozenset(
                                                [
                                                    (
                                                        TARGET_IDX,
                                                        FIRST_MOCK_PARSER,
                                                    ),
                                                ]
                                            )
                                        ),
                                        EquivalentIdSet(
                                            ids_and_source=frozenset(
                                                [
                                                    (
                                                        CONFUSING_IDX,
                                                        FIRST_MOCK_PARSER,
                                                    ),
                                                ]
                                            )
                                        ),
                                    ]
                                ),
                            )
                        ]
                    ),
                    case_sensitive=False,
                ),
            ],
            [],
            0,
            set(),
            [],
            PARSER_1_AMBIGUOUS_DATA,
            PARSER_1_AMBIGUOUS_DATA,
            False,
            id="Should not throw exception on populate databases as parser data is ambiguous and action specifies both ids",
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
    throws_curation_exception,
):
    Singleton.clear_all()
    kazu_disk_cache.clear()
    TEST_CURATIONS_PATH_PARSER_1 = tmp_path / "parser1_curated_terms.jsonl"
    TEST_CURATIONS_PATH_PARSER_2 = tmp_path / "parser2_curated_terms.jsonl"
    write_curations(path=TEST_CURATIONS_PATH_PARSER_1, terms=parser_1_curations)
    write_curations(path=TEST_CURATIONS_PATH_PARSER_2, terms=parser_2_curations)
    parser_1 = DummyParser(
        name=FIRST_MOCK_PARSER,
        entity_class="ent_type_1",
        source=FIRST_MOCK_PARSER,
        curations=load_curated_terms(path=TEST_CURATIONS_PATH_PARSER_1),
        data=parser_1_data,
    )
    parser_2 = DummyParser(
        name="second_mock_parser",
        entity_class="ent_type_2",
        source=SECOND_MOCK_PARSER,
        curations=load_curated_terms(path=TEST_CURATIONS_PATH_PARSER_2),
        data=parser_2_data,
    )
    TEST_SPAN_KEY = "my_hits"
    TEST_OUTPUT_DIR = tmp_path / "ontology_pipeline"
    if throws_curation_exception:
        with pytest.raises(CurationException):
            assemble_pipeline(
                parsers=[parser_1, parser_2],
                output_dir=TEST_OUTPUT_DIR,
                span_key=TEST_SPAN_KEY,
            )

    else:
        nlp = assemble_pipeline(
            parsers=[parser_1, parser_2],
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


def test_pipeline_build_from_parsers_alone(tmp_path):
    Singleton.clear_all()
    kazu_disk_cache.clear()
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

    doc = nlp(example_text)
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


def test_no_spacy_tokenization_update():
    # have this as a test rather than an assert because we don't want this to break things for our users -
    # it doesn't really matter to them if the tokenization is a bit outdated relative to spacy, we just
    # want to know as developers that we should look at updating.
    assert set(TOKENIZER_INFIXES) == set(
        SPACY_DEFAULT_INFIXES
    ), "Our tokenization rules are outdated to spacy's"
