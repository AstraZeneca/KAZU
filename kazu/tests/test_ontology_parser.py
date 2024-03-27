import json
from copy import deepcopy
from pathlib import Path
from typing import Optional, Literal

import pytest
from kazu.data import (
    OntologyStringResource,
    MentionConfidence,
    ParserAction,
    EquivalentIdSet,
    ParserBehaviour,
    GlobalParserActions,
    OntologyStringBehaviour,
    Synonym,
    kazu_json_converter,
)
from kazu.database.in_memory_db import SynonymDatabase
from kazu.ontology_preprocessing.base import (
    IDX,
    DEFAULT_LABEL,
    SYN,
    MAPPING_TYPE,
)
from kazu.ontology_preprocessing.curation_utils import (
    load_global_actions,
    dump_ontology_string_resources,
    CurationError,
)
from kazu.ontology_preprocessing.parsers import GeneOntologyParser
from kazu.tests.utils import DummyParser
from kazu.utils.string_normalizer import StringNormalizer
from kazu.utils.utils import Singleton

PARSER_1_NAME = "I am the target for actions"
NOOP_PARSER_NAME = "I am the result of the same parser without human curated resources"
TARGET_SYNONYM = "hello I'm injected"
# this should be split by the parser logic into two equivalent_id_sets for the same LinkingCandidate
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
    human_curated_resources: Optional[list[OntologyStringResource]] = None,
    global_actions: Optional[GlobalParserActions] = None,
    parser_data_includes_target_synonym: bool = False,
    parser_data_has_split_equiv_id_set: bool = False,
    noop_parser_curations_style: Literal["empty", "None"] = "empty",
) -> SynonymDatabase:
    Singleton.clear_all()
    curations_path = None
    if human_curated_resources is not None:
        curations_path = base_path.joinpath("curations.jsonl")
        dump_ontology_string_resources(terms=human_curated_resources, path=curations_path)

    global_actions_path = None
    if global_actions is not None:
        global_actions_path = base_path.joinpath("global_actions.json")
        with open(global_actions_path, "w") as f:
            f.writelines(json.dumps(kazu_json_converter.unstructure(global_actions)) + "\n")

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
        data=parser_data,
        entity_class=ENTITY_CLASS,
        curations_path=noop_curations_path_str,
        global_actions=None,
    )

    parser_1.populate_databases()
    noop_parser.populate_databases()

    return SynonymDatabase()


def test_should_add_synonym_term_to_parser(tmp_path):
    resource = OntologyStringResource(
        original_synonyms=frozenset(
            [
                Synonym(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    text=TARGET_SYNONYM,
                    case_sensitive=False,
                )
            ]
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_LINKING_ONLY,
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

    syn_db = setup_databases(base_path=tmp_path, human_curated_resources=[resource])

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


def test_should_modify_resource_from_parser_via_general_rule(tmp_path):
    # note, this test is similar to test_should_drop_from_parser_via_general_rule,
    # although it tests that a OntologyStringResource is also affected by a general rule
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
    resource = OntologyStringResource(
        original_synonyms=frozenset(
            [
                Synonym(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    text=TARGET_SYNONYM,
                    case_sensitive=False,
                )
            ]
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
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
    )
    syn_db = setup_databases(
        base_path=tmp_path,
        human_curated_resources=[resource],
        global_actions=global_actions,
        parser_data_includes_target_synonym=False,
        noop_parser_curations_style="None",
    )

    assert len(syn_db.get_all(PARSER_1_NAME)) + 1 == len(syn_db.get_all(NOOP_PARSER_NAME))

    assert len(syn_db.get_syns_for_id(PARSER_1_NAME, "first")) == 0


def test_should_not_add_a_term_as_id_nonexistant(tmp_path):
    override_id = "I do not exist"
    resource = OntologyStringResource(
        original_synonyms=frozenset(
            [
                Synonym(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    text=TARGET_SYNONYM,
                    case_sensitive=False,
                )
            ]
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_LINKING_ONLY,
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
    )

    syn_db = setup_databases(
        base_path=tmp_path,
        human_curated_resources=[resource],
        parser_data_includes_target_synonym=True,
    )

    assert len(syn_db.get_all(PARSER_1_NAME)) == len(syn_db.get_all(NOOP_PARSER_NAME))
    affected_term = syn_db.get(PARSER_1_NAME, StringNormalizer.normalize(TARGET_SYNONYM))
    assert len(affected_term.associated_id_sets) == 1
    modified_equivalent_ids = next(iter(affected_term.associated_id_sets)).ids
    assert override_id not in modified_equivalent_ids


def test_should_override_id_set(tmp_path):
    resource = OntologyStringResource(
        original_synonyms=frozenset(
            [
                Synonym(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    text=TARGET_SYNONYM,
                    case_sensitive=False,
                )
            ]
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_LINKING_ONLY,
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

    syn_db = setup_databases(
        base_path=tmp_path,
        human_curated_resources=[resource],
        parser_data_includes_target_synonym=True,
    )

    term_norm_lookup = resource.term_norm_for_linking(entity_class=ENTITY_CLASS)
    assert len(syn_db.get(PARSER_1_NAME, term_norm_lookup).associated_id_sets) == 1
    equiv_id_set = next(iter(syn_db.get(PARSER_1_NAME, term_norm_lookup).associated_id_sets))
    assert "first" not in equiv_id_set.ids
    assert "second" in equiv_id_set.ids


def test_should_not_add_a_synonym_term_to_db_as_one_already_exists(tmp_path):
    resource = OntologyStringResource(
        original_synonyms=frozenset(
            [
                Synonym(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    text=TARGET_SYNONYM,
                    case_sensitive=False,
                )
            ]
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_LINKING_ONLY,
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
    )

    syn_db = setup_databases(
        base_path=tmp_path,
        human_curated_resources=[resource],
        parser_data_includes_target_synonym=True,
    )

    assert len(syn_db.get_all(PARSER_1_NAME)) == len(syn_db.get_all(NOOP_PARSER_NAME))


def test_should_not_add_a_term_as_can_infer_associated_id_sets(tmp_path):
    resource = OntologyStringResource(
        original_synonyms=frozenset(
            [
                Synonym(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    text=TARGET_SYNONYM,
                    case_sensitive=False,
                )
            ]
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_LINKING_ONLY,
    )

    syn_db = setup_databases(
        base_path=tmp_path,
        human_curated_resources=[resource],
        parser_data_includes_target_synonym=True,
    )

    assert len(syn_db.get_all(PARSER_1_NAME)) == len(syn_db.get_all(NOOP_PARSER_NAME))


def test_conflicting_overrides_in_associated_id_sets(tmp_path):
    resource1 = OntologyStringResource(
        original_synonyms=frozenset(
            [
                Synonym(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    text=TARGET_SYNONYM,
                    case_sensitive=False,
                )
            ]
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_LINKING_ONLY,
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
    resource2 = OntologyStringResource(
        original_synonyms=frozenset(
            [
                Synonym(
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                    text=TARGET_SYNONYM,
                    case_sensitive=False,
                )
            ]
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_LINKING_ONLY,
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
    with pytest.raises(CurationError):
        setup_databases(
            base_path=tmp_path,
            human_curated_resources=[resource1, resource2],
            parser_data_includes_target_synonym=True,
        )


def test_gene_ontology_caching(tmp_path):
    micro_graph = """
@prefix : <https://www.example.org> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:some_ent a owl:Class ;
    rdfs:label "My cool entity" .
"""
    graph_path = tmp_path / "input_file.ttl"
    with open(graph_path, mode="w") as outf:
        outf.write(micro_graph)

    starting_cache_info = GeneOntologyParser.parse_to_graph.cache_info()
    assert (
        starting_cache_info.hits == starting_cache_info.misses == starting_cache_info.currsize == 0
    )
    GeneOntologyParser.parse_to_graph(str(graph_path))
    cache_info_post_initial_parse = GeneOntologyParser.parse_to_graph.cache_info()
    assert cache_info_post_initial_parse.currsize == cache_info_post_initial_parse.misses == 1

    # remove the file, for extra assurance it's not re-read
    graph_path.unlink()
    GeneOntologyParser.parse_to_graph(str(graph_path))
    cache_info_after_second_parse = GeneOntologyParser.parse_to_graph.cache_info()
    assert (
        cache_info_after_second_parse.hits
        == cache_info_after_second_parse.misses
        == cache_info_after_second_parse.currsize
        == 1
    )
