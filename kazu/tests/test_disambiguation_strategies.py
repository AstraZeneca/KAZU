from itertools import chain

from kazu.data.data import (
    Document,
    Mapping,
    StringMatchConfidence,
    DisambiguationConfidence,
    Entity,
    EquivalentIdAggregationStrategy,
    EquivalentIdSet,
)
from kazu.database.in_memory_db import MetadataDatabase
from kazu.steps.linking.post_processing.disambiguation.context_scoring import TfIdfScorer
from kazu.steps.linking.post_processing.disambiguation.strategies import (
    DisambiguationStrategy,
    DefinedElsewhereInDocumentDisambiguationStrategy,
    TfIdfDisambiguationStrategy,
    AnnotationLevelDisambiguationStrategy,
)
from kazu.steps.linking.post_processing.mapping_strategies.strategies import MappingFactory
from kazu.tests.utils import DummyParser
from kazu.utils.utils import Singleton


def check_ids_are_represented(
    ids_to_check: set[str],
    strategy: DisambiguationStrategy,
    doc: Document,
    parser: DummyParser,
    ents_to_tests: list[Entity],
):
    strategy.prepare(doc)
    all_id_sets: set[EquivalentIdSet] = set(
        chain.from_iterable(
            term.associated_id_sets
            for ent in ents_to_tests
            for term in ent.syn_term_to_synonym_terms.values()
        )
    )
    disambiguated_id_sets = list(
        strategy.disambiguate(id_sets=all_id_sets, document=doc, parser_name=parser.name)
    )
    assert len(disambiguated_id_sets) == len(ids_to_check)
    for idx in ids_to_check:
        assert any(idx in id_set.ids for id_set in disambiguated_id_sets)


def test_DefinedElsewhereInDocumentStrategy(set_up_p27_test_case):
    terms, parser = set_up_p27_test_case

    text1 = "p27 is often confused, but in this context it's Autoantigen p27"
    text2 = ", and definitely not CDKN1B"

    def create_doc_with_ents(ent_list: list[Entity]) -> Document:
        """
        we need s fresh document for every test, as the cache on .prepare will otherwise be full
        :param ent_list:
        :return:
        """

        doc = Document.create_simple_document(text1 + text2)
        doc.sections[0].entities.extend(ent_list)
        return doc

    autoantip27_ent = Entity.load_contiguous_entity(
        start=len(text1) - len("Autoantigen p27"),
        end=len("Autoantigen p27"),
        match="Autoantigen p27",
        entity_class="gene",
        namespace="test",
    )
    autoantip27_ent.update_terms(terms)

    p27_ent = Entity.load_contiguous_entity(
        start=0,
        end=len("p27"),
        match="p27",
        entity_class="gene",
        namespace="test",
    )
    p27_ent.update_terms(terms)

    doc = create_doc_with_ents([autoantip27_ent, p27_ent])
    strategy = DefinedElsewhereInDocumentDisambiguationStrategy(
        DisambiguationConfidence.HIGHLY_LIKELY
    )

    # first check no mappings are produced until we add the 'good' mapping info
    check_ids_are_represented(
        ids_to_check=set(), strategy=strategy, doc=doc, parser=parser, ents_to_tests=[p27_ent]
    )

    # now add a good mapping, that should be selected from the set of terms
    target_term = next(filter(lambda x: x.term_norm == "AUTOANTIGEN P 27", terms))
    target_id_set_for_good_mapping = next(
        filter(lambda x: "3" in x.ids, target_term.associated_id_sets)
    )
    target_mappings: set[Mapping] = set()
    target_mappings.update(
        MappingFactory.create_mapping_from_id_set(
            id_set=target_id_set_for_good_mapping,
            parser_name=target_term.parser_name,
            string_match_strategy="test",
            disambiguation_strategy=None,
            string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
            additional_metadata=None,
        )
    )
    autoantip27_ent.mappings.update(target_mappings)

    doc = create_doc_with_ents([autoantip27_ent, p27_ent])
    check_ids_are_represented(
        ids_to_check={"3"}, strategy=strategy, doc=doc, parser=parser, ents_to_tests=[p27_ent]
    )

    #     finally, we'll add another entity with a relevant mapping. Since two relevant entities with mappings now occur in
    #     the same context, the strategy should produce a set with two ids in it
    cdkn1b_ent = Entity.load_contiguous_entity(
        start=len(text1 + text2) - len("CDKN1B"),
        end=len(text1 + text2),
        match="CDKN1B",
        entity_class="gene",
        namespace="test",
    )
    cdkn1b_ent.update_terms(terms)

    target_term = next(filter(lambda x: x.term_norm == "CDKN1B", terms))
    target_id_set_for_good_mapping = next(
        filter(lambda x: "1" in x.ids, target_term.associated_id_sets)
    )
    target_mappings = set()
    target_mappings.update(
        MappingFactory.create_mapping_from_id_set(
            id_set=target_id_set_for_good_mapping,
            parser_name=target_term.parser_name,
            string_match_strategy="test",
            disambiguation_strategy=None,
            string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
            additional_metadata=None,
        )
    )
    cdkn1b_ent.mappings.update(target_mappings)
    doc = create_doc_with_ents([autoantip27_ent, p27_ent, cdkn1b_ent])

    check_ids_are_represented(
        ids_to_check={"3", "1"}, strategy=strategy, doc=doc, parser=parser, ents_to_tests=[p27_ent]
    )


def test_TfIdfContextStrategy(set_up_p27_test_case, mock_build_vectoriser_cache):
    # we need to clear out the scorer singleton.
    # TODO: find a better way to handle this
    # we can't call Singleton.clear_all() because we need the modifications made
    # to the databases by set_up_p27_test_case to be preserved.
    # maybe we can refactor how our test fixtures work to enable
    # clearing before the set_up_p27_test_case is run (but currently
    # that fixture is session scoped).
    if TfIdfScorer in Singleton._instances:
        Singleton._instances.pop(TfIdfScorer)
    terms, parser = set_up_p27_test_case

    text = "p27 is often confused, but in this context it's CDKN1B"
    doc = Document.create_simple_document(text)
    cdkn1b_ent = Entity.load_contiguous_entity(
        start=len(text) - len("CDKN1B"),
        end=len(text),
        match="CDKN1B",
        entity_class="gene",
        namespace="test",
    )
    doc.sections[0].entities.append(cdkn1b_ent)
    p27_ent = Entity.load_contiguous_entity(
        start=0,
        end=len("p27"),
        match="p27",
        entity_class="gene",
        namespace="test",
    )

    p27_ent.update_terms(terms)
    doc.sections[0].entities.append(p27_ent)

    strategy = TfIdfDisambiguationStrategy(
        scorer=TfIdfScorer(),
        context_threshold=0.0,
        relevant_aggregation_strategies=[EquivalentIdAggregationStrategy.NO_STRATEGY],
        confidence=DisambiguationConfidence.POSSIBLE,
    )

    check_ids_are_represented(
        ids_to_check={"1"}, strategy=strategy, doc=doc, parser=parser, ents_to_tests=[p27_ent]
    )


def test_AnnotationLevelDisambiguationStrategy(set_up_p27_test_case):

    terms, parser = set_up_p27_test_case
    text = "p27 is often confused, but in this context it's CDKN1B"
    doc = Document.create_simple_document(text)
    cdkn1b_ent = Entity.load_contiguous_entity(
        start=len(text) - len("CDKN1B"),
        end=len("CDKN1B"),
        match="CDKN1B",
        entity_class="gene",
        namespace="test",
    )
    cdkn1b_ent.update_terms(terms)
    doc.sections[0].entities.append(cdkn1b_ent)
    metadata_db = MetadataDatabase()

    # add annotation scores to the metadata_db
    all_metadata = metadata_db.get_all(parser.name)
    for idx, meta_dict in all_metadata.items():
        if idx == "1":
            meta_dict["annotation_score"] = 10
        else:
            meta_dict["annotation_score"] = 5
    strategy = AnnotationLevelDisambiguationStrategy(DisambiguationConfidence.POSSIBLE)
    check_ids_are_represented(
        ids_to_check={"1"}, strategy=strategy, doc=doc, parser=parser, ents_to_tests=[cdkn1b_ent]
    )

    # now repeat with equal scores
    for idx, meta_dict in all_metadata.items():
        if idx == "1" or idx == "3":
            meta_dict["annotation_score"] = 10
        else:
            meta_dict["annotation_score"] = 5
    check_ids_are_represented(
        ids_to_check={"1", "3"},
        strategy=strategy,
        doc=doc,
        parser=parser,
        ents_to_tests=[cdkn1b_ent],
    )
