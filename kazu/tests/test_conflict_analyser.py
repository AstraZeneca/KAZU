import pytest
from kazu.data.data import CuratedTerm, MentionForm, MentionConfidence, CuratedTermBehaviour
from kazu.ontology_preprocessing.base import CuratedTermConflictAnalyser


@pytest.mark.parametrize("autofix", [True, False])
def test_case_conflict_within_single_curation(autofix):
    case_conflicted_curation_set = {
        CuratedTerm(
            original_forms=frozenset(
                [
                    MentionForm(
                        string="hello",
                        mention_confidence=MentionConfidence.PROBABLE,
                        case_sensitive=True,
                    ),
                    MentionForm(
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
        MentionForm(
            string="hello",
            mention_confidence=MentionConfidence.PROBABLE,
            case_sensitive=True,
        ),
        MentionForm(
            string="Hello",
            mention_confidence=MentionConfidence.POSSIBLE,
            case_sensitive=False,
        ),
    ]

    case_conflicted_curation_set = {
        CuratedTerm(
            original_forms=frozenset([expected_merged_forms[0]]),
            behaviour=CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
        ),
        CuratedTerm(
            original_forms=frozenset([expected_merged_forms[1]]),
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
        next(iter(curation_report.clean_curations)).active_ner_forms()
    )


@pytest.mark.parametrize("autofix", [True, False])
def test_case_conflict_across_multiple_curations(autofix):
    expected_merged_forms = [
        MentionForm(
            string="hello",
            mention_confidence=MentionConfidence.PROBABLE,
            case_sensitive=True,
        ),
        MentionForm(
            string="Hello",
            mention_confidence=MentionConfidence.PROBABLE,
            case_sensitive=False,
        ),
    ]

    case_conflicted_curation_set = {
        CuratedTerm(
            original_forms=frozenset([expected_merged_forms[0]]),
            behaviour=CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
        ),
        CuratedTerm(
            original_forms=frozenset([expected_merged_forms[1]]),
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
        # the element of the detected_case_conflicts set will be a merged curation, so the original ones should have
        # been removed
        assert case_conflicted_curation_set not in curation_report.case_conflicts
