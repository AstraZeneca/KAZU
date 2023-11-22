import dataclasses

import pytest

from kazu.data.data import (
    Document,
    Mapping,
    StringMatchConfidence,
    Entity,
    SynonymTermWithMetrics,
)
from kazu.database.in_memory_db import SynonymDatabase
from kazu.language.string_similarity_scorers import SapbertStringSimilarityScorer
from kazu.ontology_preprocessing.base import IDX, DEFAULT_LABEL, SYN, MAPPING_TYPE
from kazu.steps.linking.post_processing.mapping_strategies.strategies import (
    ExactMatchMappingStrategy,
    SymbolMatchMappingStrategy,
    TermNormIsSubStringMappingStrategy,
    StrongMatchMappingStrategy,
    StrongMatchWithEmbeddingConfirmationStringMatchingStrategy,
)
from kazu.tests.utils import DummyParser, requires_model_pack
from kazu.utils.grouping import sort_then_group
from kazu.utils.string_normalizer import StringNormalizer


@pytest.fixture(scope="session")
def set_up_disease_mapping_test_case() -> tuple[set[SynonymTermWithMetrics], DummyParser]:

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
    terms_with_metrics = set(
        SynonymTermWithMetrics.from_synonym_term(term)
        for term in SynonymDatabase().get_all(parser.name).values()
    )
    return terms_with_metrics, parser


def check_correct_terms_selected(terms: set[SynonymTermWithMetrics], mappings: list[Mapping]):
    term_ids = set(
        (
            term.parser_name,
            idx,
        )
        for term in terms
        for id_set in term.associated_id_sets
        for idx in id_set.ids
    )
    mapping_ids = set(
        (
            mapping.parser_name,
            mapping.idx,
        )
        for mapping in mappings
    )
    assert len(term_ids.symmetric_difference(mapping_ids)) == 0


def test_ExactMatchStringMatchingStrategy(set_up_p27_test_case):
    terms, parser = set_up_p27_test_case

    text1 = "p27 is often confused"
    ent_match = "p27"
    ent_match_norm = StringNormalizer.normalize(ent_match)

    target_term = next(filter(lambda x: x.term_norm == ent_match_norm, terms))
    target_term_exact_match = dataclasses.replace(target_term, exact_match=True)
    terms.remove(target_term)
    terms.add(target_term_exact_match)
    doc = Document.create_simple_document(text1)
    p27_ent = Entity.load_contiguous_entity(
        start=0,
        end=len("p27"),
        match="p27",
        entity_class="test",  # we set this to 'test' instead of gene for consistent stringnormaliser behaviour
        namespace="test",
    )
    p27_ent.update_terms(terms)

    doc.sections[0].entities.append(p27_ent)
    strategy = ExactMatchMappingStrategy(confidence=StringMatchConfidence.HIGHLY_LIKELY)
    strategy.prepare(doc)
    mappings = list(
        strategy(
            ent_match=ent_match,
            ent_match_norm=ent_match_norm,
            document=doc,
            terms=frozenset(terms),
        )
    )

    check_correct_terms_selected({target_term}, mappings)


def test_SymbolMatchStringMatchingStrategy(set_up_p27_test_case):
    terms, parser = set_up_p27_test_case

    text1 = "PAK-2p27 is often confused"
    ent_match = "PAK-2p27"

    ent_match_norm = StringNormalizer.normalize(ent_match)

    target_term = next(filter(lambda x: x.term_norm == ent_match_norm, terms))
    doc = Document.create_simple_document(text1)
    p27_ent = Entity.load_contiguous_entity(
        start=0,
        end=len("PAK-2p27"),
        match="PAK-2p27",
        entity_class="test",
        namespace="test",
    )
    p27_ent.update_terms(terms)

    doc.sections[0].entities.append(p27_ent)
    strategy = SymbolMatchMappingStrategy(confidence=StringMatchConfidence.HIGHLY_LIKELY)
    strategy.prepare(doc)
    mappings = list(
        strategy(
            ent_match=ent_match,
            ent_match_norm=ent_match_norm,
            document=doc,
            terms=frozenset(terms),
        )
    )

    check_correct_terms_selected({target_term}, mappings)


def test_TermNormIsSubStringStringMatchingStrategy(set_up_p27_test_case):
    terms, parser = set_up_p27_test_case
    text1 = "CDKN1B gene has the wrong NER spans on it"
    ent_match = "CDKN1B gene"

    ent_match_norm = StringNormalizer.normalize(ent_match)

    target_term = next(filter(lambda x: x.term_norm == "CDKN1B", terms))

    doc = Document.create_simple_document(text1)
    p27_ent = Entity.load_contiguous_entity(
        start=0,
        end=len(ent_match),
        match=ent_match,
        entity_class="test",
        namespace="test",
    )
    p27_ent.update_terms(terms)

    doc.sections[0].entities.append(p27_ent)
    strategy = TermNormIsSubStringMappingStrategy(confidence=StringMatchConfidence.HIGHLY_LIKELY)
    strategy.prepare(doc)
    mappings = list(
        strategy(
            ent_match=ent_match,
            ent_match_norm=ent_match_norm,
            document=doc,
            terms=frozenset(terms),
        )
    )

    check_correct_terms_selected({target_term}, mappings)


@pytest.mark.parametrize("search_threshold,differential", [(100.0, 0.0), (85.0, 15.0)])
def test_StrongMatchStringMatchingStrategy(set_up_p27_test_case, search_threshold, differential):
    terms, parser = set_up_p27_test_case

    text1 = "p27 is often confused"
    ent_match = "p27"
    terms_with_scores = set()
    target_terms = set()
    for i, (_, terms) in enumerate(sort_then_group(terms, key_func=lambda x: x.associated_id_sets)):
        for term in terms:
            if i == 0:
                new_term = dataclasses.replace(term, search_score=100.0)
                terms_with_scores.add(new_term)
                target_terms.add(new_term)
            elif i == 1:
                new_term = dataclasses.replace(term, search_score=88.0)
                terms_with_scores.add(new_term)
                if differential > 0.0:
                    target_terms.add(new_term)
            else:
                new_term = dataclasses.replace(term, search_score=70.0)
                terms_with_scores.add(new_term)

    doc = Document.create_simple_document(text1)
    p27_ent = Entity.load_contiguous_entity(
        start=0,
        end=len("p27"),
        match="p27",
        entity_class="test",
        namespace="test",
    )
    p27_ent.update_terms(terms_with_scores)

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
            terms=frozenset(terms_with_scores),
        )
    )

    check_correct_terms_selected(target_terms, mappings)


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

    string_scorer = SapbertStringSimilarityScorer(model_path=kazu_test_config.SapBertHelper.path)

    terms, parser = set_up_disease_mapping_test_case
    terms_with_scores = set()
    target_terms = set()

    for term in terms:
        new_term = dataclasses.replace(term, search_score=95.0)
        terms_with_scores.add(new_term)
        if target_string in term.term_norm:
            target_terms.add(term)

    doc = Document.create_simple_document(text)
    disease_ent = Entity.load_contiguous_entity(
        start=0,
        end=len(ent_match),
        match=ent_match,
        entity_class="disease",
        namespace="test",
    )
    disease_ent.update_terms(terms_with_scores)

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
            terms=frozenset(terms_with_scores),
        )
    )

    check_correct_terms_selected(target_terms, mappings)
