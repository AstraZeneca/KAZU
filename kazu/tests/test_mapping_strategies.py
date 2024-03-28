from typing import Iterable

import pytest
from hydra.utils import instantiate

from kazu.data import (
    Document,
    Mapping,
    StringMatchConfidence,
    Entity,
    LinkingCandidate,
    LinkingMetrics,
    CandidatesToMetrics,
)
from kazu.database.in_memory_db import SynonymDatabase
from kazu.language.string_similarity_scorers import SapbertStringSimilarityScorer
from kazu.ontology_preprocessing.base import IDX, DEFAULT_LABEL, SYN, MAPPING_TYPE
from kazu.steps.linking.post_processing.mapping_strategies.strategies import (
    ExactMatchMappingStrategy,
    SymbolMatchMappingStrategy,
    SynNormIsSubStringMappingStrategy,
    StrongMatchMappingStrategy,
    StrongMatchWithEmbeddingConfirmationStringMatchingStrategy,
)
from kazu.tests.utils import DummyParser, requires_model_pack
from kazu.utils.grouping import sort_then_group
from kazu.utils.string_normalizer import StringNormalizer

pytestmark = pytest.mark.usefixtures("mock_kazu_disk_cache_on_parsers")


@pytest.fixture(scope="session")
def set_up_disease_mapping_test_case() -> tuple[CandidatesToMetrics, DummyParser]:

    dummy_data = {
        IDX: ["1", "1", "2"],
        DEFAULT_LABEL: ["Heck's disease", "Heck's disease", "Neck Disease"],
        SYN: [
            "Heck's disease",
            "Heck disease",
            "Neck Disease",
        ],
        MAPPING_TYPE: ["", "", ""],
    }
    parser = DummyParser(data=dummy_data, name="test_tfidf_parsr", source="test_tfidf_parsr")
    parser.populate_databases()
    candidates_with_metrics = {
        candidate: LinkingMetrics() for candidate in SynonymDatabase().get_all(parser.name).values()
    }
    return candidates_with_metrics, parser


def check_correct_candidates_selected(
    candidates: Iterable[LinkingCandidate], mappings: list[Mapping]
):
    candidate_ids = set(
        (
            candidate.parser_name,
            idx,
        )
        for candidate in candidates
        for id_set in candidate.associated_id_sets
        for idx in id_set.ids
    )
    mapping_ids = set(
        (
            mapping.parser_name,
            mapping.idx,
        )
        for mapping in mappings
    )
    assert len(candidate_ids.symmetric_difference(mapping_ids)) == 0


def test_ExactMatchStringMatchingStrategy(set_up_p27_test_case):
    candidates, parser = set_up_p27_test_case

    text1 = "p27 is often confused"
    ent_match = "p27"
    ent_match_norm = StringNormalizer.normalize(ent_match)

    target_candidate = next(filter(lambda x: x.synonym_norm == ent_match_norm, candidates))
    exact_match_metric = LinkingMetrics(exact_match=True)
    candidates[target_candidate] = exact_match_metric
    doc = Document.create_simple_document(text1)
    p27_ent = Entity.load_contiguous_entity(
        start=0,
        end=len("p27"),
        match="p27",
        entity_class="test",  # we set this to 'test' instead of gene for consistent stringnormaliser behaviour
        namespace="test",
    )
    p27_ent.add_or_update_linking_candidates(candidates)

    doc.sections[0].entities.append(p27_ent)
    strategy = ExactMatchMappingStrategy(confidence=StringMatchConfidence.HIGHLY_LIKELY)
    strategy.prepare(doc)
    mappings = list(
        strategy(
            ent_match=ent_match,
            ent_match_norm=ent_match_norm,
            document=doc,
            candidates=candidates,
        )
    )

    check_correct_candidates_selected({target_candidate}, mappings)


def test_SymbolMatchStringMatchingStrategy(set_up_p27_test_case):
    candidates, parser = set_up_p27_test_case

    text1 = "PAK-2p27 is often confused"
    ent_match = "PAK-2p27"

    ent_match_norm = StringNormalizer.normalize(ent_match)

    target_candidate = next(filter(lambda x: x.synonym_norm == ent_match_norm, candidates))
    doc = Document.create_simple_document(text1)
    p27_ent = Entity.load_contiguous_entity(
        start=0,
        end=len("PAK-2p27"),
        match="PAK-2p27",
        entity_class="test",
        namespace="test",
    )
    p27_ent.add_or_update_linking_candidates(candidates)

    doc.sections[0].entities.append(p27_ent)
    strategy = SymbolMatchMappingStrategy(confidence=StringMatchConfidence.HIGHLY_LIKELY)
    strategy.prepare(doc)
    mappings = list(
        strategy(
            ent_match=ent_match,
            ent_match_norm=ent_match_norm,
            document=doc,
            candidates=candidates,
        )
    )

    check_correct_candidates_selected({target_candidate}, mappings)


def test_SynNormIsSubStringStringMatchingStrategy(set_up_p27_test_case):
    candidates, parser = set_up_p27_test_case
    text1 = "CDKN1B gene has the wrong NER spans on it"
    ent_match = "CDKN1B gene"

    ent_match_norm = StringNormalizer.normalize(ent_match)

    target_candidate = next(filter(lambda x: x.synonym_norm == "CDKN1B", candidates))

    doc = Document.create_simple_document(text1)
    p27_ent = Entity.load_contiguous_entity(
        start=0,
        end=len(ent_match),
        match=ent_match,
        entity_class="test",
        namespace="test",
    )
    p27_ent.add_or_update_linking_candidates(candidates)

    doc.sections[0].entities.append(p27_ent)
    strategy = SynNormIsSubStringMappingStrategy(confidence=StringMatchConfidence.HIGHLY_LIKELY)
    strategy.prepare(doc)
    mappings = list(
        strategy(
            ent_match=ent_match,
            ent_match_norm=ent_match_norm,
            document=doc,
            candidates=candidates,
        )
    )

    check_correct_candidates_selected({target_candidate}, mappings)


@pytest.mark.parametrize("search_threshold,differential", [(100.0, 0.0), (85.0, 15.0)])
def test_StrongMatchStringMatchingStrategy(set_up_p27_test_case, search_threshold, differential):
    candidates, parser = set_up_p27_test_case

    text1 = "p27 is often confused"
    ent_match = "p27"
    candidates_with_scores = {}
    target_candidates = {}
    for i, (_, candidates) in enumerate(
        sort_then_group(candidates, key_func=lambda x: x.associated_id_sets)
    ):
        for candidate in candidates:
            if i == 0:
                metric = LinkingMetrics(search_score=100.0)
                candidates_with_scores[candidate] = metric
                target_candidates[candidate] = metric
            elif i == 1:
                metric = LinkingMetrics(search_score=88.0)
                candidates_with_scores[candidate] = metric
                if differential > 0.0:
                    target_candidates[candidate] = metric
            else:
                metric = LinkingMetrics(search_score=70.0)
                candidates_with_scores[candidate] = metric

    doc = Document.create_simple_document(text1)
    p27_ent = Entity.load_contiguous_entity(
        start=0,
        end=len("p27"),
        match="p27",
        entity_class="test",
        namespace="test",
    )
    p27_ent.add_or_update_linking_candidates(candidates)

    doc.sections[0].entities.append(p27_ent)
    strategy = StrongMatchMappingStrategy(
        confidence=StringMatchConfidence.HIGHLY_LIKELY,
        search_threshold=search_threshold,
        differential=differential,
    )

    strategy.prepare(doc)
    mappings = list(
        strategy(
            ent_match=ent_match,
            ent_match_norm=StringNormalizer.normalize(ent_match),
            document=doc,
            candidates=candidates_with_scores,
        )
    )

    check_correct_candidates_selected(target_candidates, mappings)


@pytest.mark.parametrize(
    "text,ent_match,target_string",
    [
        ("Neck disease is often confused with Heck Disease", "Neck disease", "NECK"),
        ("Heck disease is often confused with Neck Disease", "Heck disease", "HECK"),
    ],
)
@requires_model_pack
def test_StrongMatchWithEmbeddingConfirmationNormalisationStrategy(
    kazu_test_config, set_up_disease_mapping_test_case, text, ent_match, target_string
):
    sapbert_helper = instantiate(kazu_test_config.SapbertHelper)
    string_scorer = SapbertStringSimilarityScorer(sapbert=sapbert_helper)

    candidates, parser = set_up_disease_mapping_test_case
    candidates_with_scores = {}
    target_candidates = set()

    for candidate in candidates:
        metric = LinkingMetrics(search_score=95.0)
        candidates[candidate] = metric
        candidates_with_scores[candidate] = metric
        if target_string in candidate.synonym_norm:
            target_candidates.add(candidate)

    doc = Document.create_simple_document(text)
    disease_ent = Entity.load_contiguous_entity(
        start=0,
        end=len(ent_match),
        match=ent_match,
        entity_class="disease",
        namespace="test",
    )
    disease_ent.add_or_update_linking_candidates(candidates)

    doc.sections[0].entities.append(disease_ent)
    strategy = StrongMatchWithEmbeddingConfirmationStringMatchingStrategy(
        confidence=StringMatchConfidence.HIGHLY_LIKELY,
        search_threshold=90.0,
        differential=0.0,
        complex_string_scorer=string_scorer,
    )

    strategy.prepare(doc)
    mappings = list(
        strategy(
            ent_match=ent_match,
            ent_match_norm=StringNormalizer.normalize(ent_match),
            document=doc,
            candidates=candidates,
        )
    )

    check_correct_candidates_selected(target_candidates, mappings)
