import json
import logging
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
    load_global_actions,
    kazu_disk_cache,
    CurationProcessor,  # We MUST import disk cache from here in the tests, or it gets reinitialised!
)
from kazu.tests.utils import DummyParser
from kazu.utils.string_normalizer import StringNormalizer
from kazu.utils.utils import Singleton

PARSER_1_NAME = "I am the target for actions"
NOOP_PARSER_NAME = "I am the result of the same parser without curations"
TARGET_SYNONYM = "hello I'm injected"
# this should be split by the parser logic into two equivalent_id_sets for the same SynonymTerm
ID_TO_BE_REMOVED = TARGET_SYNONYM.replace(" ", "-")
DUMMY_PARSER_SOURCE = "test_parser_source"


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
should_add_synonym_term_to_parser = "should_add_synonym_term_to_parser"
should_drop_synonym_term_to_parser = "should_add_synonym_term_to_parser"
should_not_add_a_synonym_term_to_db_as_one_already_exists = (
    "should_not_add_a_synonym_term_to_db_as_one_already_exists"
)
should_fail_to_modify_terms_when_attempting_to_add_term = (
    "should_fail_to_modify_terms_when_attempting_to_add_term"
)
should_remove_an_EquivalentIdSet_from_a_synonym_term = (
    "should_remove_an_EquivalentIdSet_from_a_synonym_term"
)

should_drop_from_parser_via_general_rule = "should_drop_from_both_parsers_via_general_rule"
should_drop_curated_term_followed_by_adding_new_one = (
    "should_drop_curated_term_followed_by_adding_new_one"
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


def get_test_parser(
    test_id, curations_path, global_actions_path
) -> Tuple[DummyParser, DummyParser]:

    if should_add_synonym_term_to_parser in test_id:
        parser_1 = DummyParser(
            name=PARSER_1_NAME,
            source=DUMMY_PARSER_SOURCE,
            in_path="",
            entity_class=ENTITY_CLASS,
            curations=load_curated_terms(path=curations_path)
            if curations_path is not None
            else None,
            global_actions=load_global_actions(global_actions_path)
            if global_actions_path is not None
            else None,
        )
        noop_parser = DummyParser(
            name=NOOP_PARSER_NAME,
            source=DUMMY_PARSER_SOURCE,
            in_path="",
            entity_class=ENTITY_CLASS,
            curations=[] if curations_path is not None else None,
            global_actions=GlobalParserActions([]) if global_actions_path is not None else None,
        )
    elif should_drop_from_parser_via_general_rule in test_id:
        parser_1 = DummyParser(
            name=PARSER_1_NAME,
            source=DUMMY_PARSER_SOURCE,
            in_path="",
            data=parser_data_with_target_synonym(),
            entity_class=ENTITY_CLASS,
            curations=None,
            global_actions=load_global_actions(global_actions_path),
        )
        noop_parser = DummyParser(
            name=NOOP_PARSER_NAME,
            source=DUMMY_PARSER_SOURCE,
            in_path="",
            data=parser_data_with_target_synonym(),
            entity_class=ENTITY_CLASS,
            curations=None,
            global_actions=None,
        )
    elif any(
        rule in test_id
        for rule in {
            should_not_add_a_synonym_term_to_db_as_one_already_exists,
            should_fail_to_modify_terms_when_attempting_to_add_term,
            should_drop_curated_term_followed_by_adding_new_one,
        }
    ):
        parser_1 = DummyParser(
            name=PARSER_1_NAME,
            source=DUMMY_PARSER_SOURCE,
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
        noop_parser = DummyParser(
            name=NOOP_PARSER_NAME,
            source=DUMMY_PARSER_SOURCE,
            in_path="",
            data=parser_data_with_target_synonym(),
            entity_class=ENTITY_CLASS,
            curations=[] if curations_path is not None else None,
            global_actions=load_global_actions(
                global_actions_path
            )  # even though this is a no_op parser, we still test the global actions on it
            if global_actions_path is not None
            else None,
        )

    elif should_remove_an_EquivalentIdSet_from_a_synonym_term in test_id:
        parser_1 = DummyParserWithAggOverride(
            name=PARSER_1_NAME,
            source=DUMMY_PARSER_SOURCE,
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
        noop_parser = DummyParser(
            name=NOOP_PARSER_NAME,
            source=DUMMY_PARSER_SOURCE,
            in_path="",
            data=parser_data_with_split_equiv_id_set(),
            entity_class=ENTITY_CLASS,
            curations=[] if curations_path is not None else None,
            global_actions=GlobalParserActions([]) if global_actions_path is not None else None,
        )
    else:
        raise RuntimeError("test misconfigured")
    return parser_1, noop_parser


@pytest.mark.parametrize(
    ("curation", "global_actions"),
    [
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                actions=tuple(
                    [
                        SynonymTermAction(
                            behaviour=SynonymTermBehaviour.ADD_FOR_LINKING_ONLY,
                            associated_id_sets=frozenset(
                                [
                                    EquivalentIdSet(
                                        ids_and_source=frozenset(
                                            [
                                                (
                                                    "first",
                                                    DUMMY_PARSER_SOURCE,
                                                )
                                            ]
                                        )
                                    )
                                ]
                            ),
                        )
                    ]
                ),
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            None,
            id=should_add_synonym_term_to_parser,
        ),
        pytest.param(
            None,
            GlobalParserActions(
                actions=[
                    ParserAction(
                        behaviour=ParserBehaviour.DROP_IDS_FROM_PARSER,
                        parser_to_target_id_mappings={
                            PARSER_1_NAME: {"first"},
                        },
                    )
                ]
            ),
            id=should_drop_from_parser_via_general_rule,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                actions=tuple(
                    [
                        SynonymTermAction(
                            behaviour=SynonymTermBehaviour.ADD_FOR_LINKING_ONLY,
                            associated_id_sets=frozenset(
                                [
                                    EquivalentIdSet(
                                        ids_and_source=frozenset(
                                            [
                                                (
                                                    "first",
                                                    DUMMY_PARSER_SOURCE,
                                                )
                                            ]
                                        )
                                    )
                                ]
                            ),
                        )
                    ]
                ),
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            None,
            id=should_fail_to_modify_terms_when_attempting_to_add_term,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                actions=tuple(
                    [
                        SynonymTermAction(
                            behaviour=SynonymTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING,
                            associated_id_sets=frozenset(
                                [
                                    EquivalentIdSet(
                                        ids_and_source=frozenset(
                                            [
                                                (
                                                    "first",
                                                    DUMMY_PARSER_SOURCE,
                                                )
                                            ]
                                        )
                                    )
                                ]
                            ),
                        ),
                        SynonymTermAction(
                            behaviour=SynonymTermBehaviour.ADD_FOR_LINKING_ONLY,
                            associated_id_sets=frozenset(
                                [
                                    EquivalentIdSet(
                                        ids_and_source=frozenset(
                                            [
                                                (
                                                    "first",
                                                    DUMMY_PARSER_SOURCE,
                                                )
                                            ]
                                        )
                                    )
                                ]
                            ),
                        ),
                    ]
                ),
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            None,
            id=should_drop_curated_term_followed_by_adding_new_one,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                actions=tuple(
                    [
                        SynonymTermAction(
                            behaviour=SynonymTermBehaviour.ADD_FOR_LINKING_ONLY,
                            associated_id_sets=frozenset(
                                [
                                    EquivalentIdSet(
                                        ids_and_source=frozenset(
                                            [
                                                (
                                                    TARGET_SYNONYM,
                                                    DUMMY_PARSER_SOURCE,
                                                )
                                            ]
                                        )
                                    )
                                ]
                            ),
                        ),
                    ]
                ),
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            None,
            id=should_not_add_a_synonym_term_to_db_as_one_already_exists,
        ),
        pytest.param(
            Curation(
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                actions=tuple(
                    [
                        SynonymTermAction(
                            behaviour=SynonymTermBehaviour.DROP_ID_SET_FROM_SYNONYM_TERM,
                            associated_id_sets=frozenset(
                                [
                                    EquivalentIdSet(
                                        ids_and_source=frozenset(
                                            [
                                                (
                                                    ID_TO_BE_REMOVED,
                                                    DUMMY_PARSER_SOURCE,
                                                )
                                            ]
                                        )
                                    )
                                ]
                            ),
                        ),
                    ]
                ),
                curated_synonym=TARGET_SYNONYM,
                case_sensitive=False,
            ),
            None,
            id=should_remove_an_EquivalentIdSet_from_a_synonym_term,
        ),
    ],
)
def test_declarative_curation_logic(
    tmp_path,
    curation: Curation,
    global_actions: GlobalParserActions,
    caplog,
    request,
):
    test_id = request.node.callspec.id
    Singleton.clear_all()
    kazu_disk_cache.clear()
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
        parser_1, noop_parser = get_test_parser(test_id, curations_path, global_actions_path)

        parser_1.populate_databases()
        noop_parser.populate_databases()

        syn_db = SynonymDatabase()
        if should_add_synonym_term_to_parser in test_id:
            assert len(syn_db.get_all(PARSER_1_NAME)) == len(syn_db.get_all(NOOP_PARSER_NAME)) + 1
        elif (
            should_not_add_a_synonym_term_to_db_as_one_already_exists in test_id
            or should_fail_to_modify_terms_when_attempting_to_add_term in test_id
        ):
            assert len(syn_db.get_all(PARSER_1_NAME)) == len(syn_db.get_all(NOOP_PARSER_NAME))
        elif should_drop_from_parser_via_general_rule in test_id:
            # +2 as parser one has two entries dropped via global action
            assert len(syn_db.get_all(PARSER_1_NAME)) + 2 == len(syn_db.get_all(NOOP_PARSER_NAME))
        elif should_remove_an_EquivalentIdSet_from_a_synonym_term in test_id:
            term_norm = StringNormalizer.normalize(TARGET_SYNONYM)
            assert len(syn_db.get(PARSER_1_NAME, term_norm).associated_id_sets) == 1


def test_all_synonym_term_behaviours_handled_by_curation_processor():
    assert set(CurationProcessor.curation_apply_order) == set(SynonymTermBehaviour)
