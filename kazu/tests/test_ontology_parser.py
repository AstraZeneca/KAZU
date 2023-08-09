import json
from copy import deepcopy
from pathlib import Path
from typing import Optional, Literal

import pytest
from kazu.data.data import (
    CuratedTerm,
    MentionConfidence,
    DocumentJsonUtils,
    ParserAction,
    EquivalentIdSet,
    ParserBehaviour,
    GlobalParserActions,
    CuratedTermBehaviour,
)
from kazu.database.in_memory_db import SynonymDatabase
from kazu.ontology_preprocessing.base import (
    IDX,
    DEFAULT_LABEL,
    SYN,
    MAPPING_TYPE,
    load_global_actions,
    CurationException,
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

ENTITY_CLASS = "action_test"

pytestmark = pytest.mark.usefixtures("mock_kazu_disk_cache_on_parsers")


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
    curations: Optional[list[CuratedTerm]] = None,
    global_actions: Optional[GlobalParserActions] = None,
    parser_data_includes_target_synonym: bool = False,
    parser_data_has_split_equiv_id_set: bool = False,
    noop_parser_curations_style: Literal["empty", "None"] = "empty",
) -> SynonymDatabase:
    Singleton.clear_all()
    curations_path = None
    if curations is not None:
        curations_path = base_path.joinpath("curations.jsonl")
        with open(curations_path, "w") as f:
            for curated_term in curations:
                f.write(json.dumps(DocumentJsonUtils.obj_to_dict_repr(curated_term)) + "\n")

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
        curations_path=str(curations_path) if curations_path is not None else None,
        global_actions=load_global_actions(global_actions_path)
        if global_actions_path is not None
        else None,
    )
    noop_curations_path_str: Optional[str]
    if noop_parser_curations_style == "empty":
        noop_curations_path = base_path.joinpath("curations_noop.jsonl")
        noop_curations_path.touch()
        noop_curations_path_str = str(noop_curations_path)
    elif noop_parser_curations_style == "None":
        noop_curations_path_str = None
    else:
        raise ValueError("Invalid noop_parser_curations_style arg provided")

    noop_parser = DummyParser(
        name=NOOP_PARSER_NAME,
        source=DUMMY_PARSER_SOURCE,
        in_path="",
        data=parser_data,
        entity_class=ENTITY_CLASS,
        curations_path=noop_curations_path_str,
        global_actions=None,
    )

    parser_1.populate_databases()
    noop_parser.populate_databases()

    return SynonymDatabase()


def test_should_add_synonym_term_to_parser(tmp_path):
    curation = CuratedTerm(
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
        behaviour=CuratedTermBehaviour.ADD_FOR_LINKING_ONLY,
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

    syn_db = setup_databases(base_path=tmp_path, curations=[curation])

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


def test_should_modify_curation_from_parser_via_general_rule(tmp_path):
    # note, this test is similar to test_should_drop_from_parser_via_general_rule,
    # although it tests that a CuratedTerm is also affected by a general rule
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
    curation = CuratedTerm(
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
        behaviour=CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
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
                ),
                EquivalentIdSet(
                    ids_and_source=frozenset(
                        [
                            (
                                "second",
                                DUMMY_PARSER_SOURCE,
                            )
                        ]
                    )
                ),
            ]
        ),
        curated_synonym=TARGET_SYNONYM,
        case_sensitive=False,
    )
    syn_db = setup_databases(
        base_path=tmp_path,
        curations=[curation],
        global_actions=global_actions,
        parser_data_includes_target_synonym=False,
        noop_parser_curations_style="None",
    )

    assert len(syn_db.get_all(PARSER_1_NAME)) + 1 == len(syn_db.get_all(NOOP_PARSER_NAME))

    assert len(syn_db.get_syns_for_id(PARSER_1_NAME, "first")) == 0


def test_should_not_add_a_term_as_id_nonexistant(tmp_path):
    override_id = "I do not exist"
    curation = CuratedTerm(
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
        behaviour=CuratedTermBehaviour.ADD_FOR_LINKING_ONLY,
        associated_id_sets=frozenset(
            [
                EquivalentIdSet(
                    ids_and_source=frozenset(
                        [
                            (
                                override_id,
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
        curations=[curation],
        parser_data_includes_target_synonym=True,
    )

    assert len(syn_db.get_all(PARSER_1_NAME)) == len(syn_db.get_all(NOOP_PARSER_NAME))
    affected_term = syn_db.get(PARSER_1_NAME, StringNormalizer.normalize(TARGET_SYNONYM))
    assert len(affected_term.associated_id_sets) == 1
    modified_equivalent_ids = next(iter(affected_term.associated_id_sets)).ids
    assert override_id not in modified_equivalent_ids


def test_should_override_id_set(tmp_path):
    curation = CuratedTerm(
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
        behaviour=CuratedTermBehaviour.ADD_FOR_LINKING_ONLY,
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
        curations=[curation],
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
        behaviour=CuratedTermBehaviour.ADD_FOR_LINKING_ONLY,
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
        curations=[curation],
        parser_data_includes_target_synonym=True,
    )

    assert len(syn_db.get_all(PARSER_1_NAME)) == len(syn_db.get_all(NOOP_PARSER_NAME))


def test_should_not_add_a_term_as_can_infer_associated_id_sets(tmp_path):
    curation = CuratedTerm(
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
        behaviour=CuratedTermBehaviour.ADD_FOR_LINKING_ONLY,
        curated_synonym=TARGET_SYNONYM,
        case_sensitive=False,
    )

    syn_db = setup_databases(
        base_path=tmp_path,
        curations=[curation],
        parser_data_includes_target_synonym=True,
    )

    assert len(syn_db.get_all(PARSER_1_NAME)) == len(syn_db.get_all(NOOP_PARSER_NAME))


def test_conflicting_overrides_in_associated_id_sets(tmp_path):
    curation1 = CuratedTerm(
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
        behaviour=CuratedTermBehaviour.ADD_FOR_LINKING_ONLY,
        curated_synonym=TARGET_SYNONYM,
        case_sensitive=False,
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
    curation2 = CuratedTerm(
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
        behaviour=CuratedTermBehaviour.ADD_FOR_LINKING_ONLY,
        curated_synonym=TARGET_SYNONYM,
        case_sensitive=False,
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
    )
    with pytest.raises(CurationException):
        setup_databases(
            base_path=tmp_path,
            curations=[curation1, curation2],
            parser_data_includes_target_synonym=True,
        )
