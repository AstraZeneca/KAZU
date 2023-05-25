import json
from copy import deepcopy
from pathlib import Path
from typing import Optional, Literal, List

from kazu.data.data import (
    CuratedTerm,
    MentionConfidence,
    DocumentJsonUtils,
    ParserAction,
    EquivalentIdSet,
    ParserBehaviour,
    GlobalParserActions,
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
    kazu_disk_cache,  # We MUST import disk cache from here in the tests, or it gets reinitialised!
)
from kazu.tests.utils import DummyParser
from kazu.utils.utils import Singleton

PARSER_1_NAME = "I am the target for actions"
NOOP_PARSER_NAME = "I am the result of the same parser without curations"
TARGET_SYNONYM = "hello I'm injected"
# this should be split by the parser logic into two equivalent_id_sets for the same SynonymTerm
ID_TO_BE_REMOVED = TARGET_SYNONYM.replace(" ", "-")
DUMMY_PARSER_SOURCE = "test_parser_source"

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


def setup_databases(
    base_path: Path,
    curation: Optional[CuratedTerm] = None,
    global_actions: Optional[GlobalParserActions] = None,
    parser_data_includes_target_synonym: bool = False,
    parser_data_has_split_equiv_id_set: bool = False,
    noop_parser_curations_style: Literal["empty", "None"] = "empty",
) -> SynonymDatabase:
    Singleton.clear_all()
    kazu_disk_cache.clear()
    curations_path = None
    if curation is not None:
        curations_path = base_path.joinpath("curations.jsonl")
        with open(curations_path, "w") as f:
            f.writelines(json.dumps(DocumentJsonUtils.obj_to_dict_repr(curation)) + "\n")

    global_actions_path = None
    if global_actions is not None:
        global_actions_path = base_path.joinpath("global_actions.json")
        with open(global_actions_path, "w") as f:
            f.writelines(json.dumps(DocumentJsonUtils.obj_to_dict_repr(global_actions)) + "\n")

    if parser_data_includes_target_synonym:
        assert (
            parser_data_has_split_equiv_id_set is False
        ), "can't set both options to modify parser data"
        parser_data = parser_data_with_target_synonym()
    elif parser_data_has_split_equiv_id_set:
        parser_data = parser_data_with_split_equiv_id_set()
    else:
        # use the default dummy data
        parser_data = None

    parser_1 = DummyParser(
        name=PARSER_1_NAME,
        source=DUMMY_PARSER_SOURCE,
        in_path="",
        data=parser_data,
        entity_class=ENTITY_CLASS,
        curations=load_curated_terms(path=curations_path) if curations_path is not None else None,
        global_actions=load_global_actions(global_actions_path)
        if global_actions_path is not None
        else None,
    )

    curations: Optional[List[CuratedTerm]]
    if noop_parser_curations_style == "empty":
        curations = []
    elif noop_parser_curations_style == "None":
        curations = None
    else:
        raise ValueError("Invalid noop_parser_curations_style arg provided")

    noop_parser = DummyParser(
        name=NOOP_PARSER_NAME,
        source=DUMMY_PARSER_SOURCE,
        in_path="",
        data=parser_data,
        entity_class=ENTITY_CLASS,
        curations=curations,
        global_actions=None,
    )

    parser_1.populate_databases()
    noop_parser.populate_databases()

    return SynonymDatabase()


def test_should_add_synonym_term_to_parser(tmp_path):
    curation = CuratedTerm(
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
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
        curated_synonym=TARGET_SYNONYM,
        case_sensitive=False,
    )

    syn_db = setup_databases(base_path=tmp_path, curation=curation)

    assert len(syn_db.get_all(PARSER_1_NAME)) == len(syn_db.get_all(NOOP_PARSER_NAME)) + 1


def test_should_drop_from_parser_via_general_rule(tmp_path):
    global_actions = GlobalParserActions(
        actions=[
            ParserAction(
                behaviour=ParserBehaviour.DROP_IDS_FROM_PARSER,
                parser_to_target_id_mappings={
                    PARSER_1_NAME: {"first"},
                },
            )
        ]
    )

    syn_db = setup_databases(
        base_path=tmp_path,
        global_actions=global_actions,
        parser_data_includes_target_synonym=True,
        noop_parser_curations_style="None",
    )

    assert len(syn_db.get_all(PARSER_1_NAME)) + 2 == len(syn_db.get_all(NOOP_PARSER_NAME))


def test_should_fail_to_modify_terms_when_attempting_to_add_term(tmp_path):
    curation = CuratedTerm(
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
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
        curated_synonym=TARGET_SYNONYM,
        case_sensitive=False,
    )

    syn_db = setup_databases(
        base_path=tmp_path,
        curation=curation,
        parser_data_includes_target_synonym=True,
    )

    assert len(syn_db.get_all(PARSER_1_NAME)) == len(syn_db.get_all(NOOP_PARSER_NAME))


def test_should_drop_curated_term_followed_by_adding_new_one(tmp_path):
    curation = CuratedTerm(
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
        behaviour=SynonymTermBehaviour.ADD_FOR_LINKING_ONLY,
        associated_id_sets=frozenset(
            [
                EquivalentIdSet(
                    ids_and_source=frozenset(
                        [
                            (
                                "second",
                                DUMMY_PARSER_SOURCE,
                            )
                        ]
                    )
                )
            ]
        ),
        curated_synonym=TARGET_SYNONYM,
        case_sensitive=False,
    )

    syn_db = setup_databases(
        base_path=tmp_path,
        curation=curation,
        parser_data_includes_target_synonym=True,
    )

    term_norm_lookup = curation.term_norm_for_linking(entity_class=ENTITY_CLASS)
    assert len(syn_db.get(PARSER_1_NAME, term_norm_lookup).associated_id_sets) == 1
    equiv_id_set = next(iter(syn_db.get(PARSER_1_NAME, term_norm_lookup).associated_id_sets))
    assert "first" not in equiv_id_set.ids
    assert "second" in equiv_id_set.ids


def test_should_not_add_a_synonym_term_to_db_as_one_already_exists(tmp_path):
    curation = CuratedTerm(
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
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
        curated_synonym=TARGET_SYNONYM,
        case_sensitive=False,
    )

    syn_db = setup_databases(
        base_path=tmp_path,
        curation=curation,
        parser_data_includes_target_synonym=True,
    )

    assert len(syn_db.get_all(PARSER_1_NAME)) == len(syn_db.get_all(NOOP_PARSER_NAME))


def test_should_not_add_a_term_as_can_infer_associated_id_sets(tmp_path):
    curation = CuratedTerm(
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
        behaviour=SynonymTermBehaviour.ADD_FOR_LINKING_ONLY,
        curated_synonym=TARGET_SYNONYM,
        case_sensitive=False,
    )

    syn_db = setup_databases(
        base_path=tmp_path,
        curation=curation,
        parser_data_includes_target_synonym=True,
    )

    assert len(syn_db.get_all(PARSER_1_NAME)) == len(syn_db.get_all(NOOP_PARSER_NAME))
