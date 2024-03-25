import pytest
from kazu.data.data import CuratedTerm, Synonym, MentionConfidence, CuratedTermBehaviour
from kazu.ontology_preprocessing.base import CuratedTermConflictAnalyser


@pytest.mark.parametrize("autofix", [True, False])
def test_case_conflict_within_single_curation(autofix):
    case_conflicted_curation_set = {
        CuratedTerm(
            original_synonyms=frozenset(
                [
                    Synonym(
                        string="hello",
                        mention_confidence=MentionConfidence.PROBABLE,
                        case_sensitive=True,
                    ),
                    Synonym(
                        string="Hello",
                        mention_confidence=MentionConfidence.PROBABLE,
                        case_sensitive=False,
                    ),
                ]
            ),
            behaviour=CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
        )
    }

    conflict_analyser = CuratedTermConflictAnalyser("test", autofix=autofix)

    curation_report = conflict_analyser.verify_curation_set_integrity(case_conflicted_curation_set)

    if autofix:
        assert len(curation_report.clean_curations) == 1
        assert len(curation_report.merged_curations) == 0
        assert len(curation_report.normalisation_conflicts) == 0
        assert len(curation_report.case_conflicts) == 0
    else:
        assert len(curation_report.clean_curations) == 0
        assert len(curation_report.merged_curations) == 0
        assert len(curation_report.normalisation_conflicts) == 0
        assert case_conflicted_curation_set in curation_report.case_conflicts


@pytest.mark.parametrize("autofix", [True, False])
def test_conflict_analyser_should_merge_curations(autofix):
    expected_merged_forms = [
        Synonym(
            string="hello",
            mention_confidence=MentionConfidence.PROBABLE,
            case_sensitive=True,
        ),
        Synonym(
            string="Hello",
            mention_confidence=MentionConfidence.POSSIBLE,
            case_sensitive=False,
        ),
    ]

    case_conflicted_curation_set = {
        CuratedTerm(
            original_synonyms=frozenset([expected_merged_forms[0]]),
            behaviour=CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
        ),
        CuratedTerm(
            original_synonyms=frozenset([expected_merged_forms[1]]),
            behaviour=CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
        ),
    }

    conflict_analyser = CuratedTermConflictAnalyser("test", autofix=autofix)

    curation_report = conflict_analyser.verify_curation_set_integrity(case_conflicted_curation_set)

    assert len(curation_report.clean_curations) == 1
    assert len(curation_report.merged_curations) == 1
    assert len(curation_report.normalisation_conflicts) == 0
    assert len(curation_report.case_conflicts) == 0
    assert set(expected_merged_forms) == set(
        next(iter(curation_report.clean_curations)).active_ner_synonyms()
    )


@pytest.mark.parametrize("autofix", [True, False])
def test_case_conflict_across_multiple_curations(autofix):
    expected_merged_forms = [
        Synonym(
            string="hello",
            mention_confidence=MentionConfidence.PROBABLE,
            case_sensitive=True,
        ),
        Synonym(
            string="Hello",
            mention_confidence=MentionConfidence.PROBABLE,
            case_sensitive=False,
        ),
    ]

    case_conflicted_curation_set = {
        CuratedTerm(
            original_synonyms=frozenset([expected_merged_forms[0]]),
            behaviour=CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
        ),
        CuratedTerm(
            original_synonyms=frozenset([expected_merged_forms[1]]),
            behaviour=CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
        ),
    }

    conflict_analyser = CuratedTermConflictAnalyser("test", autofix=autofix)

    curation_report = conflict_analyser.verify_curation_set_integrity(case_conflicted_curation_set)

    if autofix:
        assert len(curation_report.clean_curations) == 1
        assert len(curation_report.merged_curations) == 1
        assert len(curation_report.normalisation_conflicts) == 0
        assert len(curation_report.case_conflicts) == 0
    else:
        assert len(curation_report.clean_curations) == 0
        assert len(curation_report.merged_curations) == 1
        assert len(curation_report.normalisation_conflicts) == 0
        assert len(curation_report.case_conflicts) == 1


@pytest.mark.parametrize("autofix", [True, False])
def test_normalisation_and_case_conflict_resolution(autofix):
    """In this test, we check that simultaneous normalisation and case conflicts are
    appropriately handled.

    The mergable_curation_set contains two terms that normalise to the same value and
    should be merged into a new curated term. However, the unmergable_curation is
    classified as symbolic by the string normaliser, and doesn't produce an equal
    normalisation value, and therefore won't be merged. The resulting new curation will
    now conflict on case with unmergable_curation, and should be resolved as per the
    value of autofix.
    """

    ner_and_linking_curation_mergeable_1 = CuratedTerm(
        original_synonyms=frozenset(
            [
                Synonym(
                    string="Estrogens, conjugated synthetic a",
                    mention_confidence=MentionConfidence.PROBABLE,
                    case_sensitive=True,
                )
            ]
        ),
        behaviour=CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
    )
    ner_and_linking_curation_mergeable_2 = CuratedTerm(
        original_synonyms=frozenset(
            [
                Synonym(
                    string="Estrogens,conjugated synthetic a",
                    mention_confidence=MentionConfidence.PROBABLE,
                    case_sensitive=True,
                )
            ]
        ),
        behaviour=CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
    )

    linking_only_curation = CuratedTerm(
        original_synonyms=frozenset(
            [
                Synonym(
                    string="Estrogens ,conjugated synthetic a",
                    mention_confidence=MentionConfidence.PROBABLE,
                    case_sensitive=True,
                )
            ]
        ),
        behaviour=CuratedTermBehaviour.ADD_FOR_LINKING_ONLY,
    )

    drop_curation = CuratedTerm(
        original_synonyms=frozenset(
            [
                Synonym(
                    string="Estrogens, conjugated synthetic a",
                    mention_confidence=MentionConfidence.PROBABLE,
                    case_sensitive=False,
                ),
            ]
        ),
        behaviour=CuratedTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING,
    )
    case_conflict_curation = CuratedTerm(
        original_synonyms=frozenset(
            [
                Synonym(
                    string="ESTROGENS, CONJUGATED SYNTHETIC A",
                    mention_confidence=MentionConfidence.PROBABLE,
                    case_sensitive=False,
                )
            ]
        ),
        behaviour=CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
    )

    # The mergeable_curation_set contains two terms that normalise to the same value and should be merged into a single curation
    # in all cases. This new curation now causes a case conflict on case_conflict_curation. If autofix=True, this is resolved.
    # if autofix = False, the conflict set appears in mergeable_curation_report.case_conflicts
    mergeable_curation_set = {
        ner_and_linking_curation_mergeable_1,
        ner_and_linking_curation_mergeable_2,
        case_conflict_curation,
    }

    conflict_analyser = CuratedTermConflictAnalyser("drug", autofix=autofix)

    mergeable_curation_report = conflict_analyser.verify_curation_set_integrity(
        mergeable_curation_set
    )
    if autofix:
        assert len(mergeable_curation_report.clean_curations) == 2
        assert len(mergeable_curation_report.merged_curations) == 1
        assert len(mergeable_curation_report.normalisation_conflicts) == 0
        assert len(mergeable_curation_report.case_conflicts) == 0
    else:
        assert len(mergeable_curation_report.clean_curations) == 0
        assert len(mergeable_curation_report.merged_curations) == 1
        assert len(mergeable_curation_report.normalisation_conflicts) == 0
        assert len(mergeable_curation_report.case_conflicts) == 1

    # The unmergeable_curation_set_1/2 contain two terms that normalise to the same value and cannot be merged into a single curation.
    # If autofix = True, the conflict is resolved via the conflict analyser resolution logic.
    # However, this resolved curation now causes a conflict on case_conflict_curation. If autofix = True, this is also resolved.
    # if autofix = False, the conflict is reported in curation_report.normalisation_conflicts. Until the normalisation conflict
    # is resolved, whether or not it will cause a case conflict with case_conflict_curation is undetermined. Therefore, case_conflict_curation
    # is reported as a clean curation.

    unmergeable_curation_set_1 = {
        ner_and_linking_curation_mergeable_1,
        linking_only_curation,
        case_conflict_curation,
    }

    unmergeable_curation_set_2 = {
        ner_and_linking_curation_mergeable_1,
        drop_curation,
        case_conflict_curation,
    }

    for conflict_set in [unmergeable_curation_set_1, unmergeable_curation_set_2]:
        curation_report = conflict_analyser.verify_curation_set_integrity(conflict_set)
        if autofix:
            assert len(curation_report.clean_curations) == 2
            assert len(curation_report.merged_curations) == 1
            assert len(curation_report.normalisation_conflicts) == 0
            assert len(curation_report.case_conflicts) == 0
        else:
            assert len(curation_report.clean_curations) == 1
            assert case_conflict_curation in curation_report.clean_curations
            assert len(curation_report.merged_curations) == 0
            assert len(curation_report.normalisation_conflicts) == 1
            conflict_set.discard(case_conflict_curation)
            assert conflict_set == next(iter(curation_report.normalisation_conflicts))
            assert len(curation_report.case_conflicts) == 0
