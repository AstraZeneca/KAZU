from typing import Iterable

from kazu.data.data import EquivalentIdSet, EquivalentIdAggregationStrategy, SynonymTerm
from kazu.modelling.language.string_similarity_scorers import (
    NGramStringSimilarityScorer,
    EntitySubtypeStringSimilarityScorer,
    NumberMatchStringSimilarityScorer,
    EntityNounModifierStringSimilarityScorer,
    RapidFuzzStringSimilarityScorer,
)
from kazu.utils.string_normalizer import StringNormalizer


def test_NGramStringSimilarityScorer():
    hit = make_term_for_scorer_test(["bowel cancer"])
    scorer = NGramStringSimilarityScorer()
    ent_match = "bowels cancer"
    assert (
        scorer(
            match=ent_match,
            match_norm=StringNormalizer.normalize(ent_match),
            term_norm=hit.term_norm,
        )
        > 0.0
    )


def test_EntitySubtypeStringSimilarityScorer():
    hit_1 = make_term_for_scorer_test(["type II diabetes", "type 2 diabetes"])
    hit_2 = make_term_for_scorer_test(["type I diabetes", "type 1 diabetes"])
    scorer = EntitySubtypeStringSimilarityScorer()
    ent_match = "diabetes, type 2"
    assert scorer(
        match=ent_match, match_norm=StringNormalizer.normalize(ent_match), term_norm=hit_1.term_norm
    )
    assert not scorer(
        match=ent_match, match_norm=StringNormalizer.normalize(ent_match), term_norm=hit_2.term_norm
    )

    # TODO: currently this test fails due to inappropriate string normalisation. ideally, we want this hit scorer
    # to also capture patters like 14C, 14D
    # hit_1 = make_term_for_scorer_test(["protein phosphatase 1 regulatory inhibitor subunit 14C",])
    # hit_2 = make_term_for_scorer_test(["protein phosphatase 1 regulatory inhibitor subunit 14D"])
    # scorer = EntitySubtypeStringSimilarityScorer()
    # ent_match = "PPP1R 14C"
    # assert scorer(match=ent_match, match_norm=StringNormalizer.normalize(ent_match), term_norm=hit_1.term_norm)
    # assert not scorer(match=ent_match, match_norm=StringNormalizer.normalize(ent_match), term_norm=hit_2.term_norm)


def test_NumberMatchStringSimilarityScorer():
    hit_1 = make_term_for_scorer_test(["MAP1LC3A"])
    hit_2 = make_term_for_scorer_test(["MAP2LC3A"])
    scorer = NumberMatchStringSimilarityScorer()
    ent_match = "MAP1LC3A gene"
    assert scorer(
        match=ent_match, match_norm=StringNormalizer.normalize(ent_match), term_norm=hit_1.term_norm
    )
    assert not scorer(
        match=ent_match, match_norm=StringNormalizer.normalize(ent_match), term_norm=hit_2.term_norm
    )


def test_EntityNounModifierStringSimilarityScorer():

    hit_1 = make_term_for_scorer_test(["CPI17-like"])
    hit_2 = make_term_for_scorer_test(["CPI17"])

    scorer = EntityNounModifierStringSimilarityScorer()
    ent_match = "CPI17 like"

    assert scorer(
        match=ent_match, match_norm=StringNormalizer.normalize(ent_match), term_norm=hit_1.term_norm
    )
    assert not scorer(
        match=ent_match, match_norm=StringNormalizer.normalize(ent_match), term_norm=hit_2.term_norm
    )

    hit_1 = make_term_for_scorer_test(["CPI17"])
    hit_2 = make_term_for_scorer_test(["CPI17 pseudogene"])

    scorer = EntityNounModifierStringSimilarityScorer()
    ent_match = "CPI17"
    assert scorer(
        match=ent_match, match_norm=StringNormalizer.normalize(ent_match), term_norm=hit_1.term_norm
    )
    assert not scorer(
        match=ent_match, match_norm=StringNormalizer.normalize(ent_match), term_norm=hit_2.term_norm
    )

    hit_1 = make_term_for_scorer_test(["epidermal growth factor receptor"])
    hit_2 = make_term_for_scorer_test(["epidermal growth factor"])

    scorer = EntityNounModifierStringSimilarityScorer()
    ent_match = "EGF receptor"
    assert scorer(
        match=ent_match, match_norm=StringNormalizer.normalize(ent_match), term_norm=hit_1.term_norm
    )
    assert not scorer(
        match=ent_match, match_norm=StringNormalizer.normalize(ent_match), term_norm=hit_2.term_norm
    )


def test_RapidFuzzStringSimilarityScorer():
    hit = make_term_for_scorer_test(["bowel cancer", "bowel carcinoma"])
    scorer = RapidFuzzStringSimilarityScorer()
    ent_match = "bowels cancer"
    assert (
        scorer(
            match=ent_match,
            match_norm=StringNormalizer.normalize(ent_match),
            term_norm=hit.term_norm,
        )
        > 0.0
    )


def make_term_for_scorer_test(synonyms: Iterable[str]) -> SynonymTerm:

    return SynonymTerm(
        terms=frozenset(synonyms),
        term_norm=StringNormalizer.normalize(next(iter(synonyms))),
        associated_id_sets=frozenset(
            (
                EquivalentIdSet(
                    ids=frozenset(["1", "2", "3"]),
                ),
            )
        ),
        parser_name="test",
        aggregated_by=EquivalentIdAggregationStrategy.NO_STRATEGY,
        is_symbolic=False,
        mapping_types=frozenset(),
    )
