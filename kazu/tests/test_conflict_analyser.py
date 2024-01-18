from kazu.data.data import CuratedTerm, MentionForm, MentionConfidence, CuratedTermBehaviour
from kazu.ontology_preprocessing.base import CuratedTermConflictAnalyser


def test_should_cause_intercuration_case_conflict():
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
                        mention_confidence=MentionConfidence.POSSIBLE,
                        case_sensitive=False,
                    ),
                ]
            ),
            behaviour=CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
        )
    }

    conflict_analyser = CuratedTermConflictAnalyser("test")

    (
        good_curations,
        merged_curations,
        detected_norm_conflicts,
        detected_case_conflicts,
    ) = conflict_analyser.verify_curation_set_integrity(case_conflicted_curation_set)
    assert len(good_curations) == 0
    assert len(merged_curations) == 0
    assert len(detected_norm_conflicts) == 0
    assert case_conflicted_curation_set in detected_case_conflicts


def test_should_merge():
    expected_merged_forms = [
        MentionForm(
            string="hello",
            mention_confidence=MentionConfidence.POSSIBLE,
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

    conflict_analyser = CuratedTermConflictAnalyser("test")

    (
        good_curations,
        merged_curations,
        detected_norm_conflicts,
        detected_case_conflicts,
    ) = conflict_analyser.verify_curation_set_integrity(case_conflicted_curation_set)
    assert len(good_curations) == 1
    assert len(merged_curations) == 1
    assert set(expected_merged_forms) == set(next(iter(good_curations)).active_ner_forms())

    assert len(detected_norm_conflicts) == 0
    assert len(detected_case_conflicts) == 0


def test_should_cause_intracuration_case_conflict():
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

    conflict_analyser = CuratedTermConflictAnalyser("test")

    (
        good_curations,
        merged_curations,
        detected_norm_conflicts,
        detected_case_conflicts,
    ) = conflict_analyser.verify_curation_set_integrity(case_conflicted_curation_set)
    assert len(good_curations) == 0
    assert len(merged_curations) == 1
    assert len(detected_norm_conflicts) == 0
    assert len(detected_case_conflicts) == 1
    # the element of the detected_case_conflicts set will be a merged curation, so the original ones should have
    # been removed
    assert case_conflicted_curation_set not in detected_case_conflicts
