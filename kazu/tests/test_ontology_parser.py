import json
import logging
import re
from copy import deepcopy
from typing import Set, Tuple, Union
from contextlib import nullcontext
import pytest
from _pytest.python_api import RaisesContext

from kazu.data.data import (
    Curation,
    MentionConfidence,
    DocumentJsonUtils,
    ParserAction,
    EquivalentIdSet,
    EquivalentIdAggregationStrategy,
    ParserBehaviour,
    AssociatedIdSets,
    GlobalParserActions,
    SynonymTermAction,
    SynonymTermBehaviour,
)
from kazu.modelling.database.in_memory_db import SynonymDatabase
from kazu.modelling.ontology_preprocessing.base import (
    IDX,
    DEFAULT_LABEL,
    SYN,
    MAPPING_TYPE,
    load_curated_terms,
    CurationException,
    load_global_actions,
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
                    (
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
                                ((ID_TO_BE_REMOVED, self.find_kb(TARGET_SYNONYM)),)
                            )
                        ),
                    )
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

should_drop_curated_term_followed_by_adding_new_one = (
    "should_drop_curated_term_followed_by_adding_new_one"
)

TERM_ADDED_REGEX = re.compile(".*SynonymTerm.*created")
GLOBAL_ACTION_MODIFICATION_REGEX = re.compile(
    ".*SynonymTerm modified count: .*, SynonymTerm dropped count: .*"
)
TERM_MODIFIED_REGEX = re.compile(".*dropped an EquivalentIdSet containing .* for key .*")
TERM_DROPPED_REXEG = re.compile(".*successfully dropped .* from database for.*")
ID_DROPPED_REXEG = re.compile(".*dropped ID .* from.*")
NO_NEED_TO_ADD_SYNONYM_TERM_REGEX = re.compile(
    ".*but term_norm <.*> already exists in synonym database.*"
)

ENTITY_CLASS = "action_test"


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


def get_test_parsers(test_id, curations_path, global_actions_path):
    if any(
        rule in test_id
        for rule in {should_add_synonym_term_to_parser_1, should_add_synonym_term_to_both_parsers}
    ):
        parser_1 = DummyParser(
            name=PARSER_1_NAME,
            in_path="",
            entity_class=ENTITY_CLASS,
            curations=load_curated_terms(path=curations_path)
            if curations_path is not None
            else None,
            global_actions=load_global_actions(global_actions_path)
            if global_actions_path is not None
            else None,
        )
        parser_2 = DummyParser(
            name=PARSER_2_NAME,
            in_path="",
            entity_class=ENTITY_CLASS,
            curations=load_curated_terms(path=curations_path)
            if curations_path is not None
            else None,
            global_actions=load_global_actions(global_actions_path)
            if global_actions_path is not None
            else None,
        )
    elif should_add_synonym_term_to_both_parser_1_and_drop_from_parser_2 in test_id:
        parser_1 = DummyParser(
            name=PARSER_1_NAME,
            in_path="",
            entity_class=ENTITY_CLASS,
            curations=load_curated_terms(path=curations_path)
            if curations_path is not None
            else None,
            global_actions=load_global_actions(global_actions_path)
            if global_actions_path is not None
            else None,
        )
        parser_2 = DummyParser(
            name=PARSER_2_NAME,
            in_path="",
            data=parser_data_with_target_synonym(),
            entity_class=ENTITY_CLASS,
            curations=load_curated_terms(path=curations_path)
            if curations_path is not None
            else None,
            global_actions=load_global_actions(global_actions_path)
            if global_actions_path is not None
            else None,
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
            entity_class=ENTITY_CLASS,
            curations=load_curated_terms(path=curations_path)
            if curations_path is not None
            else None,
            global_actions=load_global_actions(global_actions_path)
            if global_actions_path is not None
            else None,
        )
        parser_2 = DummyParser(
            name=PARSER_2_NAME,
            in_path="",
            data=parser_data_with_target_synonym(),
            entity_class=ENTITY_CLASS,
            curations=load_curated_terms(path=curations_path)
            if curations_path is not None
            else None,
            global_actions=load_global_actions(global_actions_path)
            if global_actions_path is not None
            else None,
        )
    elif should_remove_an_EquivalentIdSet_from_a_synonym_term in test_id:
        parser_1 = DummyParserWithAggOverride(
            name=PARSER_1_NAME,
            in_path="",
            data=parser_data_with_split_equiv_id_set(),
            entity_class=ENTITY_CLASS,
            curations=load_curated_terms(path=curations_path)
            if curations_path is not None
            else None,
            global_actions=load_global_actions(global_actions_path)
            if global_actions_path is not None
            else None,
        )
        parser_2 = DummyParserWithAggOverride(
            name=PARSER_2_NAME,
            in_path="",
            data=parser_data_with_split_equiv_id_set(),
            entity_class=ENTITY_CLASS,
            curations=load_curated_terms(path=curations_path)
            if curations_path is not None
            else None,
            global_actions=load_global_actions(global_actions_path)
            if global_actions_path is not None
            else None,
        )
    else:
        raise RuntimeError("test misconfigured")
    return parser_1, parser_2


@pytest.mark.parametrize(
    (
        "curation",
        "global_actions",
        "expected_log_messages",
    ),
    [
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                actions=[
                    SynonymTermAction(
                        behaviour=SynonymTermBehaviour.ADD_FOR_LINKING_ONLY,
                        parser_to_target_id_mappings={PARSER_1_NAME: {"first"}},
                        entity_class=ENTITY_CLASS,
                    )
                ],
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            None,
            [TERM_ADDED_REGEX],
            id=should_add_synonym_term_to_parser_1,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                actions=[
                    SynonymTermAction(
                        behaviour=SynonymTermBehaviour.ADD_FOR_LINKING_ONLY,
                        parser_to_target_id_mappings={
                            PARSER_1_NAME: {"first"},
                            PARSER_2_NAME: {"first"},
                        },
                        entity_class=ENTITY_CLASS,
                    )
                ],
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            None,
            [TERM_ADDED_REGEX],
            id=should_add_synonym_term_to_both_parsers,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                actions=[
                    SynonymTermAction(
                        behaviour=SynonymTermBehaviour.ADD_FOR_LINKING_ONLY,
                        parser_to_target_id_mappings={PARSER_1_NAME: {"first"}},
                        entity_class=ENTITY_CLASS,
                    ),
                    SynonymTermAction(
                        behaviour=SynonymTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING,
                        parser_to_target_id_mappings={PARSER_2_NAME: {"first"}},
                        entity_class=ENTITY_CLASS,
                    ),
                ],
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            None,
            [TERM_ADDED_REGEX, TERM_DROPPED_REXEG],
            id=should_add_synonym_term_to_both_parser_1_and_drop_from_parser_2,
        ),
        pytest.param(
            None,
            GlobalParserActions(
                actions=[
                    ParserAction(
                        behaviour=ParserBehaviour.DROP_IDS_FROM_PARSER,
                        parser_to_target_id_mappings={
                            PARSER_1_NAME: {"first"},
                            PARSER_2_NAME: {"first"},
                        },
                    )
                ]
            ),
            [GLOBAL_ACTION_MODIFICATION_REGEX],
            id=should_drop_from_both_parsers_via_general_rule,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                actions=[
                    SynonymTermAction(
                        behaviour=SynonymTermBehaviour.ADD_FOR_LINKING_ONLY,
                        parser_to_target_id_mappings={PARSER_1_NAME: {"first"}},
                        entity_class=ENTITY_CLASS,
                    )
                ],
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            None,
            [],
            id=should_raise_exception_when_attempting_to_add_term,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                actions=[
                    SynonymTermAction(
                        behaviour=SynonymTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING,
                        parser_to_target_id_mappings={PARSER_1_NAME: {"first"}},
                        entity_class=ENTITY_CLASS,
                    ),
                    SynonymTermAction(
                        behaviour=SynonymTermBehaviour.ADD_FOR_LINKING_ONLY,
                        parser_to_target_id_mappings={PARSER_1_NAME: {"first"}},
                        entity_class=ENTITY_CLASS,
                    ),
                ],
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            None,
            [TERM_DROPPED_REXEG, TERM_ADDED_REGEX],
            id=should_drop_curated_term_followed_by_adding_new_one,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                actions=[
                    SynonymTermAction(
                        behaviour=SynonymTermBehaviour.ADD_FOR_LINKING_ONLY,
                        parser_to_target_id_mappings={PARSER_1_NAME: {TARGET_SYNONYM}},
                        entity_class=ENTITY_CLASS,
                    ),
                ],
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            None,
            [NO_NEED_TO_ADD_SYNONYM_TERM_REGEX],
            id=should_not_add_a_synonym_term_to_db_as_one_already_exists,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                actions=[
                    SynonymTermAction(
                        behaviour=SynonymTermBehaviour.DROP_ID_SET_FROM_SYNONYM_TERM,
                        parser_to_target_id_mappings={PARSER_1_NAME: {ID_TO_BE_REMOVED}},
                        entity_class=ENTITY_CLASS,
                    ),
                ],
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            None,
            [TERM_MODIFIED_REGEX],
            id=should_remove_an_EquivalentIdSet_from_a_synonym_term,
        ),
    ],
)
def test_declarative_curation_logic(
    tmp_path,
    curation: Curation,
    global_actions: GlobalParserActions,
    expected_log_messages,
    caplog,
    request,
):
    test_id = request.node.callspec.id
    Singleton.clear_all()
    with caplog.at_level(logging.DEBUG):

        curations_path = None
        if curation is not None:
            curations_path = tmp_path.joinpath("curations.jsonl")
            with open(curations_path, "w") as f:
                f.writelines(json.dumps(DocumentJsonUtils.obj_to_dict_repr(curation)) + "\n")

        global_actions_path = None
        if global_actions is not None:
            global_actions_path = tmp_path.joinpath("global_actions.json")
            with open(global_actions_path, "w") as f:
                f.writelines(json.dumps(DocumentJsonUtils.obj_to_dict_repr(global_actions)) + "\n")

        # we can't parameterise the parsers as they need to load the curations from the
        # temporary directory set up using the tmp_path fixture inside the test
        parser_1, parser_2 = get_test_parsers(test_id, curations_path, global_actions_path)

        expectation: Union[RaisesContext[CurationException], nullcontext]
        if should_raise_exception_when_attempting_to_add_term in test_id:
            expectation = pytest.raises(CurationException)
        else:
            expectation = nullcontext(True)

        with expectation:
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
            assert (
                len(syn_db.get_all(PARSER_1_NAME)) == len(DummyParser.DEFAULT_DUMMY_DATA[SYN]) + 1
            )
        elif should_drop_from_both_parsers_via_general_rule in test_id:
            assert len(syn_db.get_all(PARSER_1_NAME)) == len(syn_db.get_all(PARSER_2_NAME))
            assert (
                len(syn_db.get_all(PARSER_1_NAME)) == len(DummyParser.DEFAULT_DUMMY_DATA[SYN]) - 1
            )
        elif should_remove_an_EquivalentIdSet_from_a_synonym_term in test_id:
            term_norm = StringNormalizer.normalize(TARGET_SYNONYM)
            assert len(syn_db.get(PARSER_1_NAME, term_norm).associated_id_sets) == 1
