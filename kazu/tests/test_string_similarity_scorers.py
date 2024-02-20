from collections.abc import Sequence

import pytest

from kazu.data.data import EquivalentIdSet, EquivalentIdAggregationStrategy, SynonymTerm
from kazu.language.string_similarity_scorers import (
    EntitySubtypeStringSimilarityScorer,
    NumberMatchStringSimilarityScorer,
    EntityNounModifierStringSimilarityScorer,
    RapidFuzzStringSimilarityScorer,
)
from kazu.utils.string_normalizer import StringNormalizer


@pytest.mark.parametrize(
    ("scorer", "ent_match", "matching_synonyms", "not_matching_synonyms"),
    (
        (
            EntitySubtypeStringSimilarityScorer(),
            "diabetes, type 2",
            ["type II diabetes", "type 2 diabetes"],
            ["type I diabetes", "type 1 diabetes"],
        ),
        pytest.param(
            EntitySubtypeStringSimilarityScorer(),
            "PPP1R 14C",
            [
                "protein phosphatase 1 regulatory inhibitor subunit 14C",
            ],
            ["protein phosphatase 1 regulatory inhibitor subunit 14D"],
            marks=pytest.mark.xfail(reason="inappropriate string normalisation"),
        ),
        (NumberMatchStringSimilarityScorer(), "MAP1LC3A gene", ["MAP1LC3A"], ["MAP2LC3A"]),
        (
            EntityNounModifierStringSimilarityScorer(noun_modifier_phrases=["LIKE"]),
            "CPI17 like",
            ["CPI17-like"],
            ["CPI17"],
        ),
        (
            EntityNounModifierStringSimilarityScorer(noun_modifier_phrases=["PSEUDOGENE"]),
            "CPI17",
            ["CPI17"],
            ["CPI17 pseudogene"],
        ),
        (
            EntityNounModifierStringSimilarityScorer(["RECEPTOR"]),
            "EGF receptor",
            ["epidermal growth factor receptor"],
            ["epidermal growth factor"],
        ),
    ),
)
def test_boolean_scorer(scorer, ent_match, matching_synonyms, not_matching_synonyms):
    matching_syn_term = make_term_for_scorer_test(matching_synonyms)
    not_matching_syn_term = make_term_for_scorer_test(not_matching_synonyms)

    ent_match_norm = StringNormalizer.normalize(ent_match)

    assert scorer(reference_term=ent_match_norm, query_term=matching_syn_term.term_norm)
    assert not scorer(reference_term=ent_match_norm, query_term=not_matching_syn_term.term_norm)


def test_RapidFuzzStringSimilarityScorer():
    syn_term = make_term_for_scorer_test(["bowel cancer", "bowel carcinoma"])
    scorer = RapidFuzzStringSimilarityScorer()
    ent_match = "bowels cancer"
    assert (
        scorer(
            reference_term=StringNormalizer.normalize(ent_match),
            query_term=syn_term.term_norm,
        )
        > 0.0
    )


def make_term_for_scorer_test(synonyms: Sequence[str]) -> SynonymTerm:

    return SynonymTerm(
        terms=frozenset(synonyms),
        term_norm=StringNormalizer.normalize(synonyms[0]),
        associated_id_sets=frozenset(
            (
                EquivalentIdSet(
                    ids_and_source=frozenset(
                        (
                            (
                                "1",
                                "1",
                            ),
                            (
                                "2",
                                "2",
                            ),
                            (
                                "3",
                                "3",
                            ),
                        )
                    ),
                ),
            )
        ),
        parser_name="test",
        aggregated_by=EquivalentIdAggregationStrategy.NO_STRATEGY,
        is_symbolic=False,
    )
