from typing import Iterable

from kazu.data.data import EquivalentIdSet, EquivalentIdAggregationStrategy, SynonymTerm
from kazu.modelling.language.string_similarity_scorers import (
    EntitySubtypeStringSimilarityScorer,
    NumberMatchStringSimilarityScorer,
    EntityNounModifierStringSimilarityScorer,
    RapidFuzzStringSimilarityScorer,
)
from kazu.utils.string_normalizer import StringNormalizer


def test_EntitySubtypeStringSimilarityScorer():
    syn_term_1 = make_term_for_scorer_test(["type II diabetes", "type 2 diabetes"])
    syn_term_2 = make_term_for_scorer_test(["type I diabetes", "type 1 diabetes"])
    scorer = EntitySubtypeStringSimilarityScorer()
    ent_match = "diabetes, type 2"
    assert scorer(
        reference_term=StringNormalizer.normalize(ent_match), query_term=syn_term_1.term_norm
    )
    assert not scorer(
        reference_term=StringNormalizer.normalize(ent_match), query_term=syn_term_2.term_norm
    )

    # TODO: currently this test fails due to inappropriate string normalisation. ideally, we want this scorer
    # to also capture patters like 14C, 14D
    # syn_term_1 = make_term_for_scorer_test(["protein phosphatase 1 regulatory inhibitor subunit 14C",])
    # syn_term_2 = make_term_for_scorer_test(["protein phosphatase 1 regulatory inhibitor subunit 14D"])
    # scorer = EntitySubtypeStringSimilarityScorer()
    # ent_match = "PPP1R 14C"
    # assert scorer(reference_term=StringNormalizer.normalize(ent_match), query_term=syn_term_1.term_norm)
    # assert not scorer(reference_term=StringNormalizer.normalize(ent_match), query_term=syn_term_2.term_norm)


def test_NumberMatchStringSimilarityScorer():
    syn_term_1 = make_term_for_scorer_test(["MAP1LC3A"])
    syn_term_2 = make_term_for_scorer_test(["MAP2LC3A"])
    scorer = NumberMatchStringSimilarityScorer()
    ent_match = "MAP1LC3A gene"
    assert scorer(
        reference_term=StringNormalizer.normalize(ent_match), query_term=syn_term_1.term_norm
    )
    assert not scorer(
        reference_term=StringNormalizer.normalize(ent_match), query_term=syn_term_2.term_norm
    )


def test_EntityNounModifierStringSimilarityScorer():

    syn_term_1 = make_term_for_scorer_test(["CPI17-like"])
    syn_term_2 = make_term_for_scorer_test(["CPI17"])

    scorer = EntityNounModifierStringSimilarityScorer()
    ent_match = "CPI17 like"

    assert scorer(
        reference_term=StringNormalizer.normalize(ent_match), query_term=syn_term_1.term_norm
    )
    assert not scorer(
        reference_term=StringNormalizer.normalize(ent_match), query_term=syn_term_2.term_norm
    )

    syn_term_1 = make_term_for_scorer_test(["CPI17"])
    syn_term_2 = make_term_for_scorer_test(["CPI17 pseudogene"])

    scorer = EntityNounModifierStringSimilarityScorer()
    ent_match = "CPI17"
    assert scorer(
        reference_term=StringNormalizer.normalize(ent_match), query_term=syn_term_1.term_norm
    )
    assert not scorer(
        reference_term=StringNormalizer.normalize(ent_match), query_term=syn_term_2.term_norm
    )

    syn_term_1 = make_term_for_scorer_test(["epidermal growth factor receptor"])
    syn_term_2 = make_term_for_scorer_test(["epidermal growth factor"])

    scorer = EntityNounModifierStringSimilarityScorer()
    ent_match = "EGF receptor"
    assert scorer(
        reference_term=StringNormalizer.normalize(ent_match), query_term=syn_term_1.term_norm
    )
    assert not scorer(
        reference_term=StringNormalizer.normalize(ent_match), query_term=syn_term_2.term_norm
    )


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
