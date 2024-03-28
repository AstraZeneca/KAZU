import pytest
from kazu.data import (
    OntologyStringResource,
    Synonym,
    MentionConfidence,
    OntologyStringBehaviour,
)
from kazu.ontology_preprocessing.base import OntologyStringConflictAnalyser


@pytest.mark.parametrize("autofix", [True, False])
def test_case_conflict_within_single_resource(autofix):
    case_conflicted_resources = {
        OntologyStringResource(
            original_synonyms=frozenset(
                [
                    Synonym(
                        text="hello",
                        mention_confidence=MentionConfidence.PROBABLE,
                        case_sensitive=True,
                    ),
                    Synonym(
                        text="Hello",
                        mention_confidence=MentionConfidence.PROBABLE,
                        case_sensitive=False,
                    ),
                ]
            ),
            behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
        )
    }

    conflict_analyser = OntologyStringConflictAnalyser("test", autofix=autofix)

    curation_report = conflict_analyser.verify_resource_set_integrity(case_conflicted_resources)

    if autofix:
        assert len(curation_report.clean_resources) == 1
        assert len(curation_report.merged_resources) == 0
        assert len(curation_report.normalisation_conflicts) == 0
        assert len(curation_report.case_conflicts) == 0
    else:
        assert len(curation_report.clean_resources) == 0
        assert len(curation_report.merged_resources) == 0
        assert len(curation_report.normalisation_conflicts) == 0
        assert case_conflicted_resources in curation_report.case_conflicts


@pytest.mark.parametrize("autofix", [True, False])
def test_conflict_analyser_should_merge_resources(autofix):
    expected_merged_synonyms = [
        Synonym(
            text="hello",
            mention_confidence=MentionConfidence.PROBABLE,
            case_sensitive=True,
        ),
        Synonym(
            text="Hello",
            mention_confidence=MentionConfidence.POSSIBLE,
            case_sensitive=False,
        ),
    ]

    case_conflicted_resources = {
        OntologyStringResource(
            original_synonyms=frozenset([expected_merged_synonyms[0]]),
            behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
        ),
        OntologyStringResource(
            original_synonyms=frozenset([expected_merged_synonyms[1]]),
            behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
        ),
    }

    conflict_analyser = OntologyStringConflictAnalyser("test", autofix=autofix)

    curation_report = conflict_analyser.verify_resource_set_integrity(case_conflicted_resources)

    assert len(curation_report.clean_resources) == 1
    assert len(curation_report.merged_resources) == 1
    assert len(curation_report.normalisation_conflicts) == 0
    assert len(curation_report.case_conflicts) == 0
    assert set(expected_merged_synonyms) == set(
        next(iter(curation_report.clean_resources)).active_ner_synonyms()
    )


@pytest.mark.parametrize("autofix", [True, False])
def test_case_conflict_across_multiple_resources(autofix):
    expected_merged_synonyms = [
        Synonym(
            text="hello",
            mention_confidence=MentionConfidence.PROBABLE,
            case_sensitive=True,
        ),
        Synonym(
            text="Hello",
            mention_confidence=MentionConfidence.PROBABLE,
            case_sensitive=False,
        ),
    ]

    case_conflicted_resources = {
        OntologyStringResource(
            original_synonyms=frozenset([expected_merged_synonyms[0]]),
            behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
        ),
        OntologyStringResource(
            original_synonyms=frozenset([expected_merged_synonyms[1]]),
            behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
        ),
    }

    conflict_analyser = OntologyStringConflictAnalyser("test", autofix=autofix)

    curation_report = conflict_analyser.verify_resource_set_integrity(case_conflicted_resources)

    if autofix:
        assert len(curation_report.clean_resources) == 1
        assert len(curation_report.merged_resources) == 1
        assert len(curation_report.normalisation_conflicts) == 0
        assert len(curation_report.case_conflicts) == 0
    else:
        assert len(curation_report.clean_resources) == 0
        assert len(curation_report.merged_resources) == 1
        assert len(curation_report.normalisation_conflicts) == 0
        assert len(curation_report.case_conflicts) == 1


@pytest.mark.parametrize("autofix", [True, False])
def test_normalisation_and_case_conflict_resolution(autofix):
    """In this test, we check that simultaneous normalisation and case conflicts are
    appropriately handled.

    The mergeable_resources contains two resources that normalise to the same value and
    should be merged into a new resource. However, the unmergable_resource is classified
    as symbolic by the string normaliser, and doesn't produce an equal normalisation
    value, and therefore won't be merged. The resulting new resource will now conflict
    on case with unmergable_resource, and should be resolved as per the value of
    autofix.
    """

    ner_and_linking_resource_mergeable_1 = OntologyStringResource(
        original_synonyms=frozenset(
            [
                Synonym(
                    text="Estrogens, conjugated synthetic a",
                    mention_confidence=MentionConfidence.PROBABLE,
                    case_sensitive=True,
                )
            ]
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
    )
    ner_and_linking_resource_mergeable_2 = OntologyStringResource(
        original_synonyms=frozenset(
            [
                Synonym(
                    text="Estrogens,conjugated synthetic a",
                    mention_confidence=MentionConfidence.PROBABLE,
                    case_sensitive=True,
                )
            ]
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
    )

    linking_only_resource = OntologyStringResource(
        original_synonyms=frozenset(
            [
                Synonym(
                    text="Estrogens ,conjugated synthetic a",
                    mention_confidence=MentionConfidence.PROBABLE,
                    case_sensitive=True,
                )
            ]
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_LINKING_ONLY,
    )

    drop_resource = OntologyStringResource(
        original_synonyms=frozenset(
            [
                Synonym(
                    text="Estrogens, conjugated synthetic a",
                    mention_confidence=MentionConfidence.PROBABLE,
                    case_sensitive=False,
                ),
            ]
        ),
        behaviour=OntologyStringBehaviour.DROP_FOR_LINKING,
    )
    case_conflict_resource = OntologyStringResource(
        original_synonyms=frozenset(
            [
                Synonym(
                    text="ESTROGENS, CONJUGATED SYNTHETIC A",
                    mention_confidence=MentionConfidence.PROBABLE,
                    case_sensitive=False,
                )
            ]
        ),
        behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
    )

    # The mergeable_resources contains two resources that normalise to the same value and should be merged into a single resource
    # in all cases. This new resource now causes a case conflict on case_conflict_resource. If autofix=True, this is resolved.
    # if autofix = False, the conflict set appears in mergeable_curation_report.case_conflicts
    mergeable_resources = {
        ner_and_linking_resource_mergeable_1,
        ner_and_linking_resource_mergeable_2,
        case_conflict_resource,
    }

    conflict_analyser = OntologyStringConflictAnalyser("drug", autofix=autofix)

    mergeable_curation_report = conflict_analyser.verify_resource_set_integrity(mergeable_resources)
    if autofix:
        assert len(mergeable_curation_report.clean_resources) == 2
        assert len(mergeable_curation_report.merged_resources) == 1
        assert len(mergeable_curation_report.normalisation_conflicts) == 0
        assert len(mergeable_curation_report.case_conflicts) == 0
    else:
        assert len(mergeable_curation_report.clean_resources) == 0
        assert len(mergeable_curation_report.merged_resources) == 1
        assert len(mergeable_curation_report.normalisation_conflicts) == 0
        assert len(mergeable_curation_report.case_conflicts) == 1

    # The unmergeable_resources_1/2 contain two resources that normalise to the same value and cannot be merged into a single resource.
    # If autofix = True, the conflict is resolved via the conflict analyser resolution logic.
    # However, this resolved resource now causes a conflict on case_conflict_resource. If autofix = True, this is also resolved.
    # if autofix = False, the conflict is reported in curation_report.normalisation_conflicts. Until the normalisation conflict
    # is resolved, whether or not it will cause a case conflict with case_conflict_resource is undetermined. Therefore, case_conflict_resource
    # is reported as a clean resource.

    unmergeable_resources_1 = {
        ner_and_linking_resource_mergeable_1,
        linking_only_resource,
        case_conflict_resource,
    }

    unmergeable_resources_2 = {
        ner_and_linking_resource_mergeable_1,
        drop_resource,
        case_conflict_resource,
    }

    for conflict_set in [unmergeable_resources_1, unmergeable_resources_2]:
        curation_report = conflict_analyser.verify_resource_set_integrity(conflict_set)
        if autofix:
            assert len(curation_report.clean_resources) == 2
            assert len(curation_report.merged_resources) == 1
            assert len(curation_report.normalisation_conflicts) == 0
            assert len(curation_report.case_conflicts) == 0
        else:
            assert len(curation_report.clean_resources) == 1
            assert case_conflict_resource in curation_report.clean_resources
            assert len(curation_report.merged_resources) == 0
            assert len(curation_report.normalisation_conflicts) == 1
            conflict_set.discard(case_conflict_resource)
            assert conflict_set == next(iter(curation_report.normalisation_conflicts))
            assert len(curation_report.case_conflicts) == 0
