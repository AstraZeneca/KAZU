import dataclasses

from kazu.data import OntologyStringBehaviour, MentionConfidence, Synonym, OntologyStringResource
from kazu.krt.resource_manager import ResourceManager
from kazu.krt.string_editor.utils import ResourceConflictManager
from kazu.krt.resource_discrepancy_editor.utils import ResourceDiscrepancyManger
from kazu.tests.utils import DummyParser


def init_test_resource_manager() -> ResourceManager:
    p1 = DummyParser(name="test_parser1")
    p1.curations_path = p1.ontology_auto_generated_resources_set_path
    p1.populate_databases()
    p2 = DummyParser(name="test_parser2")
    p1.curations_path = p1.ontology_auto_generated_resources_set_path
    p2.populate_databases()
    parsers = [p1, p2]
    rm = ResourceManager(parsers)
    return rm


def init_test_string_conflict_manager() -> ResourceConflictManager:

    conflict1_p1 = OntologyStringResource(
        original_synonyms=frozenset(
            {Synonym(text="4", case_sensitive=False, mention_confidence=MentionConfidence.PROBABLE)}
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
        alternative_synonyms=frozenset(),
        associated_id_sets=None,
        autocuration_results=None,
        comment=None,
    )
    conflict1_p2 = OntologyStringResource(
        original_synonyms=frozenset(
            {Synonym(text="4", case_sensitive=True, mention_confidence=MentionConfidence.PROBABLE)}
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
        alternative_synonyms=frozenset(),
        associated_id_sets=None,
        autocuration_results=None,
        comment=None,
    )
    conflict2_p1 = OntologyStringResource(
        original_synonyms=frozenset(
            {
                Synonym(
                    text="two", case_sensitive=False, mention_confidence=MentionConfidence.PROBABLE
                )
            }
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
        alternative_synonyms=frozenset(),
        associated_id_sets=None,
        autocuration_results=None,
        comment=None,
    )
    conflict2_p2 = OntologyStringResource(
        original_synonyms=frozenset(
            {
                Synonym(
                    text="two", case_sensitive=True, mention_confidence=MentionConfidence.PROBABLE
                )
            }
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
        alternative_synonyms=frozenset(),
        associated_id_sets=None,
        autocuration_results=None,
        comment=None,
    )
    p1 = DummyParser(
        name="test_parser1",
        curations_injections=set([conflict1_p1, conflict1_p2, conflict2_p1, conflict2_p2]),
    )
    p1.populate_databases()
    p2 = DummyParser(
        name="test_parser2",
        curations_injections=set([conflict1_p1, conflict1_p2, conflict2_p1, conflict2_p2]),
    )
    p2.populate_databases()
    rm = ResourceManager([p1, p2])
    scm = ResourceConflictManager(manager=rm)
    return scm


def init_discrepancy_manager() -> ResourceDiscrepancyManger:
    d1 = OntologyStringResource(
        original_synonyms=frozenset(
            {
                Synonym(
                    text="one", case_sensitive=False, mention_confidence=MentionConfidence.PROBABLE
                ),
                Synonym(
                    text="one-", case_sensitive=False, mention_confidence=MentionConfidence.PROBABLE
                ),
            }
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
        alternative_synonyms=frozenset(),
        associated_id_sets=None,
        autocuration_results=None,
        comment=None,
    )
    d2 = OntologyStringResource(
        original_synonyms=frozenset(
            {
                Synonym(
                    text="two", case_sensitive=True, mention_confidence=MentionConfidence.PROBABLE
                ),
                Synonym(
                    text="two-", case_sensitive=True, mention_confidence=MentionConfidence.PROBABLE
                ),
            }
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
        alternative_synonyms=frozenset(),
        associated_id_sets=None,
        autocuration_results=None,
        comment=None,
    )
    p1 = DummyParser(
        name="test_parser1",
        curations_injections=set([d1, d2]),
    )
    p1.populate_databases()
    rm = ResourceManager([p1])
    dm = ResourceDiscrepancyManger(parser_name="test_parser1", manager=rm)
    return dm


def test_resource_manager_sync(mock_kazu_disk_cache_on_parsers):
    rm = init_test_resource_manager()
    for parser in rm.parsers.values():
        old_resources = parser.populate_metadata_db_and_resolve_string_resources()[
            1
        ].final_conflict_report.clean_resources
        for old_resource in old_resources:
            new_resource = dataclasses.replace(
                old_resource, behaviour=OntologyStringBehaviour.DROP_FOR_LINKING
            )
            rm.sync_resources(
                original_resource=old_resource, new_resource=new_resource, parser_name=parser.name
            )
            assert old_resource not in rm.resource_to_parsers
            assert new_resource in rm.resource_to_parsers


def test_string_conflict_manager_sync(mock_kazu_disk_cache_on_parsers):
    scm = init_test_string_conflict_manager()
    new_resources: set[OntologyStringResource] = set()
    assert len(scm.unresolved_conflicts) == 2
    for conflict in scm.unresolved_conflicts.values():
        conflict.batch_resolve(optimistic=True)
        scm.sync_resources_for_resolved_resource_conflict_and_find_new_conflicts(conflict)
        for resolution_dict in conflict.parser_to_resource_to_resolution.values():
            for resource in resolution_dict.values():
                if resource is not None:
                    new_resources.add(resource)
    assert len(scm.unresolved_conflicts) == 0
    assert new_resources.issubset(scm.manager.resource_to_parsers)


def test_discrepancy_manager_sync(mock_kazu_disk_cache_on_parsers):
    dm = init_discrepancy_manager()
    new_resources: set[OntologyStringResource] = set()
    assert len(dm.unresolved_discrepancies) == 2
    for discrepancy in list(dm.unresolved_discrepancies.values()):
        new_resource = discrepancy.auto_resolve()
        assert new_resource is not None
        new_resources.add(new_resource)
        # index is always 0 as is recalculated after every commit
        dm.commit(discrepancy.human_resource, new_resource, 0)
    assert new_resources.issubset(dm.manager.resource_to_parsers)
