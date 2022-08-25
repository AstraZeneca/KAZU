import dataclasses
from typing import Tuple, Set, List

import pytest
from hydra.utils import instantiate

from kazu.data.data import (
    Document,
    Mapping,
    LinkRanks,
    Entity,
    SynonymTermWithMetrics,
)
from kazu.steps.linking.post_processing.string_matching.strategies import (
    ExactMatchStringMatchingStrategy,
    SymbolMatchStringMatchingStrategy,
    TermNormIsSubStringStringMatchingStrategy,
    DefinedElsewhereInDocumentStringMatchingStrategy,
    StrongMatchStringMatchingStrategy,
    StrongMatchWithEmbeddingConfirmationStringMatchingStrategy,
    MappingFactory,
)
from kazu.tests.utils import DummyParser, make_dummy_parser, requires_model_pack
from kazu.utils.grouping import sort_then_group
from kazu.utils.string_normalizer import StringNormalizer


def check_correct_terms_selected(terms: Set[SynonymTermWithMetrics], mappings: List[Mapping]):
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


@pytest.fixture(scope="session")
def populate_databases() -> Tuple[DummyParser, DummyParser]:

    parser1 = DummyParser("")
    parser2 = make_dummy_parser(
        in_path="", data=DummyParser.DUMMY_DATA, name="test_parser2", source="test_parser2"
    )
    for parser in [parser1, parser2]:
        parser.populate_databases()
    return parser1, parser2


def test_ExactMatchStringMatchingStrategy(set_up_p27_test_case):
    terms, parser = set_up_p27_test_case

    text1 = "p27 is often confused"
    ent_match = "p27"

    target_term = next(filter(lambda x: x.term_norm == "P27", terms))
    target_term_exact_match = dataclasses.replace(target_term, exact_match=True)
    terms.remove(target_term)
    terms.add(target_term_exact_match)
    doc = Document.create_simple_document(text1)
    p27_ent = Entity.load_contiguous_entity(
        start=0,
        end=len("p27"),
        match="p27",
        entity_class="gene",
        namespace="test",
    )
    p27_ent.update_terms(terms)

    doc.sections[0].entities.append(p27_ent)
    strategy = ExactMatchStringMatchingStrategy(confidence=LinkRanks.HIGHLY_LIKELY)
    strategy.prepare(doc)
    mappings = list(
        strategy(
            ent_match=ent_match,
            ent_match_norm=StringNormalizer.normalize(ent_match),
            document=doc,
            terms=frozenset(terms),
        )
    )

    check_correct_terms_selected({target_term}, mappings)


def test_SymbolMatchStringMatchingStrategy(set_up_p27_test_case):
    terms, parser = set_up_p27_test_case

    text1 = "PAK-2p27 is often confused"
    ent_match = "PAK-2p27"

    target_term = next(filter(lambda x: x.term_norm == "PAK 2P27", terms))
    doc = Document.create_simple_document(text1)
    p27_ent = Entity.load_contiguous_entity(
        start=0,
        end=len("PAK-2p27"),
        match="PAK-2p27",
        entity_class="gene",
        namespace="test",
    )
    p27_ent.update_terms(terms)

    doc.sections[0].entities.append(p27_ent)
    strategy = SymbolMatchStringMatchingStrategy(confidence=LinkRanks.HIGHLY_LIKELY)
    strategy.prepare(doc)
    mappings = list(
        strategy(
            ent_match=ent_match,
            ent_match_norm=StringNormalizer.normalize(ent_match),
            document=doc,
            terms=frozenset(terms),
        )
    )

    check_correct_terms_selected({target_term}, mappings)


def test_TermNormIsSubStringStringMatchingStrategy(set_up_p27_test_case):
    terms, parser = set_up_p27_test_case
    target_term = next(filter(lambda x: x.term_norm == "CDKN1B", terms))
    text1 = "CDKN1B gene has the wrong NER spans on it"
    ent_match = "CDKN1B gene"
    doc = Document.create_simple_document(text1)
    p27_ent = Entity.load_contiguous_entity(
        start=0,
        end=len(ent_match),
        match=ent_match,
        entity_class="gene",
        namespace="test",
    )
    p27_ent.update_terms(terms)

    doc.sections[0].entities.append(p27_ent)
    strategy = TermNormIsSubStringStringMatchingStrategy(confidence=LinkRanks.HIGHLY_LIKELY)
    strategy.prepare(doc)
    mappings = list(
        strategy(
            ent_match=ent_match,
            ent_match_norm=StringNormalizer.normalize(ent_match),
            document=doc,
            terms=frozenset(terms),
        )
    )

    check_correct_terms_selected({target_term}, mappings)


def test_DefinedElsewhereInDocumentStringMatchingStrategy(set_up_p27_test_case):
    terms, parser = set_up_p27_test_case
    target_term = next(filter(lambda x: x.term_norm == "CDKN1B", terms))
    text1 = "p27 gene is also known as CDKN1B"
    ent_match = "p27"
    doc = Document.create_simple_document(text1)
    p27_ent = Entity.load_contiguous_entity(
        start=0,
        end=len(ent_match),
        match=ent_match,
        entity_class="gene",
        namespace="test",
    )
    p27_ent.update_terms(terms)
    cdkn1b_ent = Entity.load_contiguous_entity(
        start=len(text1) - len("CDKN1B"),
        end=len(text1),
        match="CDKN1B",
        entity_class="gene",
        namespace="test",
    )

    mappings = MappingFactory.create_mapping_from_id_set(
        next(iter(target_term.associated_id_sets)),
        parser_name=parser.name,
        strategy="test",
        confidence=LinkRanks.HIGHLY_LIKELY,
    )
    cdkn1b_ent.mappings.update(mappings)
    doc.sections[0].entities.append(cdkn1b_ent)

    strategy = DefinedElsewhereInDocumentStringMatchingStrategy(confidence=LinkRanks.HIGHLY_LIKELY)
    strategy.prepare(doc)
    mappings = list(
        strategy(
            ent_match=ent_match,
            ent_match_norm=StringNormalizer.normalize(ent_match),
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
        entity_class="gene",
        namespace="test",
    )
    p27_ent.update_terms(terms_with_scores)

    doc.sections[0].entities.append(p27_ent)
    strategy = StrongMatchStringMatchingStrategy(
        confidence=LinkRanks.HIGHLY_LIKELY,
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

    sapbert_string_scorer = instantiate(kazu_test_config.StringScorers.sapbert)

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
        confidence=LinkRanks.HIGHLY_LIKELY,
        search_threshold=90.0,
        differential=0.0,
        complex_string_scorer=sapbert_string_scorer,
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
