from typing import Optional, List

import pytest

from azner.steps.linking.link_ensembling import EnsembleEntityLinkingStep, LinkRanks
from azner.data.data import (
    SimpleDocument,
    Entity,
    Mapping,
    LINK_SCORE,
    NAMESPACE,
    LINK_CONFIDENCE,
    PROCESSING_EXCEPTION,
)

LINKING_THRESHOLDS = {
    "noisy_linker": 80.0,
    "good_linker": 95.0,
}


@pytest.fixture(params=[1, 3])
def perform_test(request):

    keep_top_n = request.param

    def _perform_test(
        mappings: List[Mapping], target_best_mapping: Mapping, expected_confidence: str
    ):

        step = EnsembleEntityLinkingStep(
            [], keep_top_n=keep_top_n, linker_score_thresholds=LINKING_THRESHOLDS
        )
        doc = SimpleDocument("hello")
        entity = Entity(namespace="test", start=0, end=1, match="hello", entity_class="test")
        entity.metadata.mappings = mappings
        doc.sections[0].entities = [entity]
        result, _ = step([doc])
        assert result[0].metadata.get(PROCESSING_EXCEPTION, None) is None
        result_entities = result[0].get_entities()
        result_mappings = result_entities[0].metadata.mappings
        assert len(result_mappings) <= keep_top_n
        found_best_mapping = result_mappings[0]
        assert found_best_mapping.idx == target_best_mapping.idx
        assert found_best_mapping.metadata[LINK_CONFIDENCE] == expected_confidence

    return _perform_test


def make_mapping(
    score: float, linker_namespace: str, default_label: str, syn: Optional[str], idx: str
) -> Mapping:
    return Mapping(
        start=0,
        end=1,
        idx=idx,
        mapping_type=["test"],
        source="test",
        metadata={
            "default_label": default_label,
            LINK_SCORE: score,
            NAMESPACE: linker_namespace,
            "syn": syn,
        },
    )


def test_best_score_link_ensembling(perform_test):
    bad_mappings = [
        make_mapping(
            score=0.5,
            linker_namespace="noisy_linker",
            default_label="bad_match",
            syn="goodbye",
            idx="bad_match",
        ),
        make_mapping(
            score=0.81,
            linker_namespace="noisy_linker",
            default_label="bad_match",
            syn="goodbye",
            idx="bad_match",
        ),
    ]

    # case 1: match on string overlap
    target_best_mapping = make_mapping(
        score=60.0,  # <- should match here
        linker_namespace="good_linker",
        default_label="blaaah",
        syn="goodbye",
        idx="correct",
    )
    mappings = bad_mappings + [target_best_mapping]
    perform_test(
        mappings,
        target_best_mapping,
        expected_confidence=LinkRanks.LOW_CONFIDENCE.value,
    )


def test_query_contained_in_hits_link_ensembling(perform_test):
    bad_mappings = [
        make_mapping(
            score=0.5,
            linker_namespace="noisy_linker",
            default_label="bad_match",
            syn="goodbye",
            idx="bad_match",
        ),
        make_mapping(
            score=0.81,
            linker_namespace="noisy_linker",
            default_label="bad_match",
            syn="goodbye",
            idx="bad_match",
        ),
    ]

    # case 1: match on string overlap
    target_best_mapping = make_mapping(
        score=60.0,
        linker_namespace="good_linker",
        default_label="hello to you",  # <- should match here
        syn="goodbye",
        idx="correct",
    )
    mappings = bad_mappings + [target_best_mapping]
    perform_test(
        mappings,
        target_best_mapping,
        expected_confidence=LinkRanks.MEDIUM_CONFIDENCE.value,
    )

    target_best_mapping = make_mapping(
        score=60.0,
        linker_namespace="good_linker",
        default_label="gahh",
        syn="hello to you",  # <- should match here
        idx="correct",
    )
    mappings = bad_mappings + [target_best_mapping]
    perform_test(
        mappings,
        target_best_mapping,
        expected_confidence=LinkRanks.MEDIUM_CONFIDENCE.value,
    )


def test_similarly_ranked_link_ensembling(perform_test):
    bad_mappings = [
        make_mapping(
            score=0.5,
            linker_namespace="noisy_linker",
            default_label="bad_match",
            syn="goodbye",
            idx="bad_match",
        ),
        make_mapping(
            score=0.81,
            linker_namespace="noisy_linker",
            default_label="bad_match",
            syn="goodbye",
            idx="bad_match",
        ),
    ]
    # case 1: match on ensemble of good links
    target_ok_mapping_1 = make_mapping(
        score=77.0,  # <- scores are below thresholds, but independent method of average_linker_2 suggests this is a good link
        linker_namespace="good_linker",
        default_label="nice",
        syn="goodbye",
        idx="correct",
    )
    target_ok_mapping_2 = make_mapping(
        score=75.0,
        linker_namespace="noisy_linker",
        default_label="nice",
        syn="goodbye",
        idx="correct",
    )

    mappings = bad_mappings + [target_ok_mapping_1, target_ok_mapping_2]
    perform_test(
        mappings,
        target_ok_mapping_1,
        expected_confidence=LinkRanks.MEDIUM_HIGH_CONFIDENCE.value,
    )


def test_filter_scores_link_ensembling(perform_test):
    bad_mappings = [
        make_mapping(
            score=0.5,
            linker_namespace="noisy_linker",
            default_label="bad_match",
            syn="goodbye",
            idx="bad_match",
        ),
        make_mapping(
            score=0.81,
            linker_namespace="noisy_linker",
            default_label="bad_match",
            syn="goodbye",
            idx="bad_match",
        ),
    ]

    # case 1: match on score threshold
    target_best_mapping = make_mapping(
        score=99.0,  # <- should match here
        linker_namespace="good_linker",
        default_label="nice",
        syn="goodbye",
        idx="correct",
    )
    mappings = bad_mappings + [target_best_mapping]
    perform_test(
        mappings,
        target_best_mapping,
        expected_confidence=LinkRanks.MEDIUM_HIGH_CONFIDENCE.value,
    )


def test_exact_match_link_ensembling(perform_test):
    bad_mappings = [
        make_mapping(
            score=0.5,
            linker_namespace="noisy_linker",
            default_label="bad_match",
            syn="goodbye",
            idx="bad_match",
        ),
        make_mapping(
            score=0.81,
            linker_namespace="noisy_linker",
            default_label="bad_match",
            syn="goodbye",
            idx="bad_match",
        ),
    ]

    # case 1: match on default label
    target_best_mapping = make_mapping(
        score=99.0,
        linker_namespace="good_linker",
        default_label="hello",  # <- should match here
        syn="goodbye",
        idx="correct",
    )
    mappings = bad_mappings + [target_best_mapping]
    perform_test(
        mappings,
        target_best_mapping,
        expected_confidence=LinkRanks.HIGH_CONFIDENCE.value,
    )

    # case 2: match on syn
    target_best_mapping = make_mapping(
        score=99.0,
        linker_namespace="good_linker",
        default_label="some_id",
        syn="hello",  # <- should match here
        idx="correct",
    )
    mappings = bad_mappings + [target_best_mapping]
    perform_test(
        mappings,
        target_best_mapping,
        expected_confidence=LinkRanks.HIGH_CONFIDENCE.value,
    )

    # case 3: match on perfect score
    target_best_mapping = make_mapping(
        score=100.0,  # <- should match here
        linker_namespace="good_linker",
        default_label="some_id",
        syn="some_id",
        idx="correct",
    )
    mappings = bad_mappings + [target_best_mapping]
    perform_test(
        mappings,
        target_best_mapping,
        expected_confidence=LinkRanks.HIGH_CONFIDENCE.value,
    )
