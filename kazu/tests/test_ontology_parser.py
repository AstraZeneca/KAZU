import json
import logging
import re
from copy import deepcopy
from typing import Set, Tuple

import pytest

from kazu.data.data import (
    Curation,
    MentionConfidence,
    DocumentJsonUtils,
    ParserAction,
    EquivalentIdSet,
    EquivalentIdAggregationStrategy,
    ParserBehaviour,
    AssociatedIdSets,
)
from kazu.modelling.database.in_memory_db import SynonymDatabase
from kazu.modelling.ontology_preprocessing.base import (
    IDX,
    DEFAULT_LABEL,
    SYN,
    MAPPING_TYPE,
    load_curated_terms,
    CurationException,
)
from kazu.tests.utils import DummyParser
from kazu.utils.string_normalizer import StringNormalizer
from kazu.utils.utils import Singleton

PARSER_1_NAME = "I am the target for actions"
PARSER_2_NAME = "I should (mostly) not be affected by actions"

TARGET_SYNONYM = "hello I'm injected"
# this should be split by the parser logic into two equivalent_id_sets for the same SynonymTerm
ID_TO_BE_REMOVED = TARGET_SYNONYM.replace(" ", "-")


class DummyParserWithAggOverride(DummyParser):
    def score_and_group_ids(
        self,
        id_and_source: Set[Tuple[str, str]],
        is_symbolic: bool,
        original_syn_set: Set[str],
    ) -> Tuple[AssociatedIdSets, EquivalentIdAggregationStrategy]:
        if TARGET_SYNONYM in original_syn_set:
            return (
                frozenset(
                    [
                        EquivalentIdSet(
                            ids_and_source=frozenset(
                                [
                                    (
                                        TARGET_SYNONYM,
                                        self.find_kb(TARGET_SYNONYM),
                                    )
                                ]
                            )
                        ),
                        EquivalentIdSet(
                            ids_and_source=frozenset(
                                [
                                    (ID_TO_BE_REMOVED, self.find_kb(TARGET_SYNONYM)),
                                ]
                            )
                        ),
                    ]
                ),
                EquivalentIdAggregationStrategy.CUSTOM,
            )
        else:
            return super().score_and_group_ids(id_and_source, is_symbolic, original_syn_set)


# test ids
should_add_synonym_term_to_parser_1 = "should_add_synonym_term_to_parser_1"
should_add_synonym_term_to_both_parsers = "should_add_synonym_term_to_both_parsers"
should_add_synonym_term_to_both_parser_1_and_drop_from_parser_2 = (
    "should_add_synonym_term_to_both_parser_1_and_drop_from_parser_2"
)
should_drop_from_both_parsers_via_general_rule = "should_drop_from_both_parsers_via_general_rule"
should_not_add_a_synonym_term_to_db_as_one_already_exists = (
    "should_not_add_a_synonym_term_to_db_as_one_already_exists"
)
should_raise_exception_when_attempting_to_add_term = (
    "should_raise_exception_when_attempting_to_add_term"
)
should_remove_an_EquivalentIdSet_from_a_synonym_term = (
    "should_remove_an_EquivalentIdSet_from_a_synonym_term"
)

should_remove_an_id_from_equivalent_id_set = "should_remove_an_id_from_equivalent_id_set"

should_drop_curated_term_followed_by_adding_new_one = (
    "should_drop_curated_term_followed_by_adding_new_one"
)

TERM_ADDED_REGEX = re.compile(".*SynonymTerm.*created")
TERM_MODIFIED_OR_DROPPED_REGEX = re.compile(".*modified .* and dropped .* SynonymTerms containing")
TERM_MODIFIED_REGEX = re.compile(".*dropped an EquivalentIdSet containing .* for key .*")
TERM_DROPPED_REXEG = re.compile(".*successfully dropped .* from database for.*")
ID_DROPPED_REXEG = re.compile(".*dropped ID .* from.*")
NO_NEED_TO_ADD_SYNONYM_TERM_REGEX = re.compile(
    ".*but term_norm <.*> already exists in synonym database.*"
)


def parser_data_with_target_synonym():
    parser_data = deepcopy(DummyParser.DEFAULT_DUMMY_DATA)
    parser_data[IDX].append(TARGET_SYNONYM)
    parser_data[DEFAULT_LABEL].append(TARGET_SYNONYM)
    parser_data[SYN].append(TARGET_SYNONYM)
    parser_data[MAPPING_TYPE].append(TARGET_SYNONYM)
    return parser_data


def parser_data_with_split_equiv_id_set():
    parser_data = deepcopy(DummyParser.DEFAULT_DUMMY_DATA)
    parser_data[IDX].append(TARGET_SYNONYM)
    parser_data[DEFAULT_LABEL].append(TARGET_SYNONYM)
    parser_data[SYN].append(TARGET_SYNONYM)
    parser_data[MAPPING_TYPE].append(TARGET_SYNONYM)
    parser_data[IDX].append(ID_TO_BE_REMOVED)
    parser_data[DEFAULT_LABEL].append(ID_TO_BE_REMOVED)
    parser_data[SYN].append(TARGET_SYNONYM)
    parser_data[MAPPING_TYPE].append(ID_TO_BE_REMOVED)
    return parser_data


def get_test_parsers(test_id, path):
    if any(
        rule in test_id
        for rule in {should_add_synonym_term_to_parser_1, should_add_synonym_term_to_both_parsers}
    ):
        parser_1 = DummyParser(
            name=PARSER_1_NAME,
            in_path="",
            entity_class="injection_test",
            curations=load_curated_terms(path=path),
        )
        parser_2 = DummyParser(
            name=PARSER_2_NAME,
            in_path="",
            entity_class="injection_test",
            curations=load_curated_terms(path=path),
        )
    elif should_add_synonym_term_to_both_parser_1_and_drop_from_parser_2 in test_id:
        parser_1 = DummyParser(
            name=PARSER_1_NAME,
            in_path="",
            entity_class="injection_test",
            curations=load_curated_terms(path=path),
        )
        parser_2 = DummyParser(
            name=PARSER_2_NAME,
            in_path="",
            data=parser_data_with_target_synonym(),
            entity_class="injection_test",
            curations=load_curated_terms(path=path),
        )
    elif any(
        rule in test_id
        for rule in {
            should_drop_from_both_parsers_via_general_rule,
            should_not_add_a_synonym_term_to_db_as_one_already_exists,
            should_raise_exception_when_attempting_to_add_term,
            should_drop_curated_term_followed_by_adding_new_one,
        }
    ):
        parser_1 = DummyParser(
            name=PARSER_1_NAME,
            in_path="",
            data=parser_data_with_target_synonym(),
            entity_class="injection_test",
            curations=load_curated_terms(path=path),
        )
        parser_2 = DummyParser(
            name=PARSER_2_NAME,
            in_path="",
            data=parser_data_with_target_synonym(),
            entity_class="injection_test",
            curations=load_curated_terms(path=path),
        )
    elif should_remove_an_EquivalentIdSet_from_a_synonym_term in test_id:
        parser_1 = DummyParserWithAggOverride(
            name=PARSER_1_NAME,
            in_path="",
            data=parser_data_with_split_equiv_id_set(),
            entity_class="injection_test",
            curations=load_curated_terms(path=path),
        )
        parser_2 = DummyParserWithAggOverride(
            name=PARSER_2_NAME,
            in_path="",
            data=parser_data_with_split_equiv_id_set(),
            entity_class="injection_test",
            curations=load_curated_terms(path=path),
        )
    elif should_remove_an_id_from_equivalent_id_set in test_id:
        parser_1 = DummyParserWithAggOverride(
            name=PARSER_1_NAME,
            in_path="",
            data=parser_data_with_split_equiv_id_set(),
            entity_class="injection_test",
            curations=load_curated_terms(path=path),
        )
        parser_2 = DummyParserWithAggOverride(
            name=PARSER_2_NAME,
            in_path="",
            data=parser_data_with_split_equiv_id_set(),
            entity_class="injection_test",
            curations=load_curated_terms(path=path),
        )
    else:
        raise RuntimeError("test misconfigured")
    return parser_1, parser_2


@pytest.mark.parametrize(
    (
        "curation",
        "expected_log_messages",
    ),
    [
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                ner_actions=[],
                parser_actions=[
                    ParserAction(
                        behaviour=ParserBehaviour.ADD,
                        parser_to_target_id_mapping={PARSER_1_NAME: "first"},
                    )
                ],
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            [TERM_ADDED_REGEX],
            id=should_add_synonym_term_to_parser_1,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                ner_actions=[],
                parser_actions=[
                    ParserAction(
                        behaviour=ParserBehaviour.ADD,
                        parser_to_target_id_mapping={PARSER_1_NAME: "first", PARSER_2_NAME: "first"},
                    )
                ],
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            [TERM_ADDED_REGEX],
            id=should_add_synonym_term_to_both_parsers,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                ner_actions=[],
                parser_actions=[
                    ParserAction(
                        behaviour=ParserBehaviour.ADD,
                        parser_to_target_id_mapping={PARSER_1_NAME: "first"},
                    ),
                    ParserAction(
                        behaviour=ParserBehaviour.DROP_SYNONYM_TERM_FROM_PARSER,
                        parser_to_target_id_mapping={PARSER_2_NAME: None},
                    ),
                ],
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            [TERM_ADDED_REGEX, TERM_DROPPED_REXEG],
            id=should_add_synonym_term_to_both_parser_1_and_drop_from_parser_2,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                ner_actions=[],
                parser_actions=[
                    ParserAction(
                        behaviour=ParserBehaviour.DROP_SYNONYM_TERM_FROM_PARSER,
                        parser_to_target_id_mapping={},
                    )
                ],
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            [TERM_DROPPED_REXEG],
            id=should_drop_from_both_parsers_via_general_rule,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                ner_actions=[],
                parser_actions=[
                    ParserAction(
                        behaviour=ParserBehaviour.ADD,
                        parser_to_target_id_mapping={PARSER_1_NAME: "first"},
                    ),
                ],
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            [],
            id=should_raise_exception_when_attempting_to_add_term,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                ner_actions=[],
                parser_actions=[
                    ParserAction(
                        behaviour=ParserBehaviour.DROP_SYNONYM_TERM_FROM_PARSER,
                        parser_to_target_id_mapping={},
                    ),
                    ParserAction(
                        behaviour=ParserBehaviour.ADD,
                        parser_to_target_id_mapping={PARSER_1_NAME: "first"},
                    ),
                ],
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            [TERM_DROPPED_REXEG, TERM_ADDED_REGEX],
            id=should_drop_curated_term_followed_by_adding_new_one,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                ner_actions=[],
                parser_actions=[
                    ParserAction(
                        behaviour=ParserBehaviour.ADD,
                        parser_to_target_id_mapping={PARSER_1_NAME: TARGET_SYNONYM},
                    ),
                ],
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            [NO_NEED_TO_ADD_SYNONYM_TERM_REGEX],
            id=should_not_add_a_synonym_term_to_db_as_one_already_exists,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                ner_actions=[],
                parser_actions=[
                    ParserAction(
                        behaviour=ParserBehaviour.DROP_ID_SET_FROM_SYNONYM_TERM,
                        parser_to_target_id_mapping={PARSER_1_NAME: ID_TO_BE_REMOVED},
                    )
                ],
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            [TERM_MODIFIED_REGEX],
            id=should_remove_an_EquivalentIdSet_from_a_synonym_term,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                ner_actions=[],
                parser_actions=[
                    ParserAction(
                        behaviour=ParserBehaviour.DROP_ID_FROM_PARSER,
                        parser_to_target_id_mapping={PARSER_1_NAME: ID_TO_BE_REMOVED},
                    )
                ],
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            [ID_DROPPED_REXEG],
            id=should_remove_an_id_from_equivalent_id_set,
        ),
    ],
)
def test_declarative_curation_logic(
    tmp_path, curation: Curation, expected_log_messages, caplog, request
):
    test_id = request.node.callspec.id
    Singleton.clear_all()
    with caplog.at_level(logging.INFO):
        path = tmp_path.joinpath("injections.jsonl")
        with open(path, "w") as f:
            f.writelines(json.dumps(DocumentJsonUtils.obj_to_dict_repr(curation)) + "\n")

        # we can't parameterise the parsers as they
        parser_1, parser_2 = get_test_parsers(test_id, path)

        if should_raise_exception_when_attempting_to_add_term in test_id:
            with pytest.raises(CurationException):
                parser_1.populate_databases()
                parser_2.populate_databases()
        else:
            parser_1.populate_databases()
            parser_2.populate_databases()
        assert all(x.search(caplog.text) is not None for x in expected_log_messages)

        syn_db = SynonymDatabase()
        if (
            should_add_synonym_term_to_parser_1 in test_id
            or should_add_synonym_term_to_both_parser_1_and_drop_from_parser_2 in test_id
        ):
            assert len(syn_db.get_all(PARSER_1_NAME)) == len(syn_db.get_all(PARSER_2_NAME)) + 1
        elif (
            should_add_synonym_term_to_both_parsers in test_id
            or should_not_add_a_synonym_term_to_db_as_one_already_exists in test_id
        ):
            assert len(syn_db.get_all(PARSER_1_NAME)) == len(syn_db.get_all(PARSER_2_NAME))
            assert len(syn_db.get_all(PARSER_1_NAME)) == len(DummyParser.DEFAULT_DUMMY_DATA[SYN]) + 1
        elif should_drop_from_both_parsers_via_general_rule in test_id:
            assert len(syn_db.get_all(PARSER_1_NAME)) == len(syn_db.get_all(PARSER_2_NAME))
            assert len(syn_db.get_all(PARSER_1_NAME)) == len(DummyParser.DEFAULT_DUMMY_DATA[SYN])
        elif should_remove_an_EquivalentIdSet_from_a_synonym_term in test_id:
            term_norm = StringNormalizer.normalize(TARGET_SYNONYM)
            assert len(syn_db.get(PARSER_1_NAME, term_norm).associated_id_sets) == 1
        elif should_remove_an_id_from_equivalent_id_set in test_id:
            term_norm = StringNormalizer.normalize(TARGET_SYNONYM)
            assert len(syn_db.get(PARSER_1_NAME, term_norm).associated_id_sets) == 1
