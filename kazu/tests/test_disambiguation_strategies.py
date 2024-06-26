import json
from itertools import chain

import pytest
from hydra.utils import instantiate
from kazu.data import (
    Document,
    Mapping,
    StringMatchConfidence,
    DisambiguationConfidence,
    Entity,
    EquivalentIdAggregationStrategy,
    EquivalentIdSet,
    LinkingMetrics,
)
from kazu.database.in_memory_db import MetadataDatabase, SynonymDatabase
from kazu.ontology_preprocessing.base import IDX, DEFAULT_LABEL, SYN, MAPPING_TYPE
from kazu.steps.linking.post_processing.disambiguation.context_scoring import (
    TfIdfScorer,
    GildaTfIdfScorer,
)
from kazu.steps.linking.post_processing.disambiguation.strategies import (
    DisambiguationStrategy,
    DefinedElsewhereInDocumentDisambiguationStrategy,
    TfIdfDisambiguationStrategy,
    AnnotationLevelDisambiguationStrategy,
    PreferDefaultLabelMatchDisambiguationStrategy,
    PreferNearestEmbeddingToDefaultLabelDisambiguationStrategy,
    GildaTfIdfDisambiguationStrategy,
)
from kazu.steps.linking.post_processing.mapping_strategies.strategies import MappingFactory
from kazu.tests.utils import DummyParser, requires_model_pack
from kazu.utils.utils import Singleton

pytestmark = pytest.mark.usefixtures("mock_kazu_disk_cache_on_parsers")


def check_ids_are_represented(
    ids_to_check: set[str],
    strategy: DisambiguationStrategy,
    doc: Document,
    parser: DummyParser,
    entity_to_test: Entity,
):
    strategy.prepare(doc)
    all_id_sets: set[EquivalentIdSet] = set(
        chain.from_iterable(
            candidate.associated_id_sets for candidate in entity_to_test.linking_candidates
        )
    )
    disambiguated_id_sets = list(
        strategy.disambiguate(
            id_sets=all_id_sets,
            document=doc,
            parser_name=parser.name,
            ent_match=entity_to_test.match,
            ent_match_norm=entity_to_test.match_norm,
        )
    )
    assert len(disambiguated_id_sets) == len(ids_to_check)
    for idx in ids_to_check:
        assert any(idx in id_set.ids for id_set in disambiguated_id_sets)


def test_DefinedElsewhereInDocumentStrategy(set_up_p27_test_case):
    candidates, parser = set_up_p27_test_case

    text1 = "p27 is often confused, but in this context it's Autoantigen p27"
    text2 = ", and definitely not CDKN1B"

    def create_doc_with_ents(ent_list: list[Entity]) -> Document:
        """We need a fresh document for every test, as the cache on .prepare will
        otherwise be full.

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
    autoantip27_ent.add_or_update_linking_candidates(candidates)

    p27_ent = Entity.load_contiguous_entity(
        start=0,
        end=len("p27"),
        match="p27",
        entity_class="gene",
        namespace="test",
    )
    p27_ent.add_or_update_linking_candidates(candidates)

    doc = create_doc_with_ents([autoantip27_ent, p27_ent])
    strategy = DefinedElsewhereInDocumentDisambiguationStrategy(
        DisambiguationConfidence.HIGHLY_LIKELY
    )

    # first check no mappings are produced until we add the 'good' mapping info
    check_ids_are_represented(
        ids_to_check=set(), strategy=strategy, doc=doc, parser=parser, entity_to_test=p27_ent
    )

    # now add a good mapping, that should be selected from the set of candidates
    target_candidate = next(filter(lambda x: x.synonym_norm == "AUTOANTIGEN P 27", candidates))
    target_id_set_for_good_mapping = next(
        filter(lambda x: "3" in x.ids, target_candidate.associated_id_sets)
    )
    target_mappings: set[Mapping] = set()
    target_mappings.update(
        MappingFactory.create_mapping_from_id_set(
            id_set=target_id_set_for_good_mapping,
            parser_name=target_candidate.parser_name,
            string_match_strategy="test",
            disambiguation_strategy=None,
            string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
            additional_metadata=None,
        )
    )
    autoantip27_ent.mappings.update(target_mappings)

    doc = create_doc_with_ents([autoantip27_ent, p27_ent])
    check_ids_are_represented(
        ids_to_check={"3"}, strategy=strategy, doc=doc, parser=parser, entity_to_test=p27_ent
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
    cdkn1b_ent.add_or_update_linking_candidates(candidates)

    target_candidate = next(filter(lambda x: x.synonym_norm == "CDKN1B", candidates))
    target_id_set_for_good_mapping = next(
        filter(lambda x: "1" in x.ids, target_candidate.associated_id_sets)
    )
    target_mappings = set()
    target_mappings.update(
        MappingFactory.create_mapping_from_id_set(
            id_set=target_id_set_for_good_mapping,
            parser_name=target_candidate.parser_name,
            string_match_strategy="test",
            disambiguation_strategy=None,
            string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
            additional_metadata=None,
        )
    )
    cdkn1b_ent.mappings.update(target_mappings)
    doc = create_doc_with_ents([autoantip27_ent, p27_ent, cdkn1b_ent])

    check_ids_are_represented(
        ids_to_check={"3", "1"}, strategy=strategy, doc=doc, parser=parser, entity_to_test=p27_ent
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
    candidates, parser = set_up_p27_test_case

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

    p27_ent.add_or_update_linking_candidates(candidates)
    doc.sections[0].entities.append(p27_ent)

    strategy = TfIdfDisambiguationStrategy(
        scorer=TfIdfScorer(),
        context_threshold=0.0,
        relevant_aggregation_strategies=[EquivalentIdAggregationStrategy.NO_STRATEGY],
        confidence=DisambiguationConfidence.POSSIBLE,
    )

    check_ids_are_represented(
        ids_to_check={"1"}, strategy=strategy, doc=doc, parser=parser, entity_to_test=p27_ent
    )


def test_AnnotationLevelDisambiguationStrategy(set_up_p27_test_case):

    candidates, parser = set_up_p27_test_case
    text = "p27 is often confused, but in this context it's CDKN1B"
    doc = Document.create_simple_document(text)
    cdkn1b_ent = Entity.load_contiguous_entity(
        start=len(text) - len("CDKN1B"),
        end=len("CDKN1B"),
        match="CDKN1B",
        entity_class="gene",
        namespace="test",
    )
    cdkn1b_ent.add_or_update_linking_candidates(candidates)
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
        ids_to_check={"1"}, strategy=strategy, doc=doc, parser=parser, entity_to_test=cdkn1b_ent
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
        entity_to_test=cdkn1b_ent,
    )


def test_PreferDefaultLabelMatchDisambiguationStrategy():
    dummy_data = {
        IDX: ["1", "1", "1", "2", "2", "2", "3", "3", "3"],
        DEFAULT_LABEL: [
            "CDKN1B",
            "CDKN1B",
            "CDKN1B",
            "PAK2",
            "PAK2",
            "PAK2",
            "ZNRD2",
            "ZNRD2",
            "ZNRD2",
        ],
        SYN: [
            "cyclin-dependent kinase inhibitor 1B (p27, Kip1)",
            "CDKN1B",
            "p27",
            "PAK-2p27",
            "p27",
            "PAK2",
            "CDKN1B",  # this label is ambiguous, but it's not the default one, so IDX 2 and 3 should be eliminated
            "ZNRD2",
            "p27",
        ],
        MAPPING_TYPE: ["", "", "", "", "", "", "", "", ""],
    }
    parser = DummyParser(
        data=dummy_data, name="test_default_label_strategy", source="test_default_label_strategy"
    )
    parser.populate_databases()

    text = "CDKN1B is confused in this test,but with this strategy we want the id where CDKN1B is the default label"
    doc = Document.create_simple_document(text)
    cdkn1b_ent = Entity.load_contiguous_entity(
        start=0,
        end=len(text),
        match="CDKN1B",
        entity_class=parser.entity_class,
        namespace="test",
    )
    for candidate in SynonymDatabase().get_all(parser.name).values():
        cdkn1b_ent.add_or_update_linking_candidate(candidate, LinkingMetrics())
    doc.sections[0].entities.append(cdkn1b_ent)

    strategy = PreferDefaultLabelMatchDisambiguationStrategy(
        confidence=DisambiguationConfidence.PROBABLE,
    )

    check_ids_are_represented(
        ids_to_check={"1"}, strategy=strategy, doc=doc, parser=parser, entity_to_test=cdkn1b_ent
    )


@requires_model_pack
def test_GildaTfIdfContextStrategy(
    set_up_p27_test_case, tmp_path, override_kazu_test_config, mock_build_gilda_vectoriser_cache
):

    Singleton._instances.pop(GildaTfIdfScorer, None)

    candidates, parser = set_up_p27_test_case
    json_path = tmp_path.joinpath("p27_disamb_test.json")
    data: dict[str, dict[str, str]] = {
        parser.name: {
            "1": """Cyclin-dependent kinase inhibitor 1B (p27Kip1) is an enzyme inhibitor that in humans is encoded by the CDKN1B gene.[5]
It encodes a protein which belongs to the Cip/Kip family of cyclin dependent kinase (Cdk) inhibitor proteins.
The encoded protein binds to and prevents the activation of cyclin E-CDK2 or cyclin D-CDK4 complexes, and thus controls the cell cycle progression at G1.
It is often referred to as a cell cycle inhibitor protein because its major function is to stop or slow down the cell division cycle."""
        }
    }

    for candidate in candidates:
        for equiv_id_set in candidate.associated_id_sets:
            for idx in equiv_id_set.ids:
                if idx != "1":
                    data[parser.name][idx] = "this is not relevant"

    with json_path.open(mode="w") as f:
        json.dump(data, f)

    cfg = override_kazu_test_config(
        overrides=[
            f"DisambiguationStrategies.gene.1.scorer.contexts_path={str(json_path)}",
        ]
    )

    strategy = instantiate(cfg.DisambiguationStrategies.gene[1])
    assert isinstance(strategy, GildaTfIdfDisambiguationStrategy)

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

    p27_ent.add_or_update_linking_candidates(candidates)
    doc.sections[0].entities.append(p27_ent)

    check_ids_are_represented(
        ids_to_check={"1"}, strategy=strategy, doc=doc, parser=parser, entity_to_test=p27_ent
    )


@requires_model_pack
def test_PreferNearestEmbeddingToDefaultLabelDisambiguationStrategy(kazu_test_config):

    dummy_data = {
        IDX: ["1", "2", "3"],
        DEFAULT_LABEL: [
            "breast cancer",
            "lung cancer",
            "mouth cancer",
        ],
        SYN: [
            "breast carcinoma",
            "breast carcinoma",
            "breast carcinoma",
        ],
        MAPPING_TYPE: ["", "", ""],
    }
    parser = DummyParser(
        data=dummy_data,
        name="test_prefer_nearest_embedding",
        source="test_prefer_nearest_embedding",
        entity_class="disease",
    )
    parser.populate_databases()
    candidates_with_metrics = {
        candidate: LinkingMetrics() for candidate in SynonymDatabase().get_all(parser.name).values()
    }

    text = "breast carcinoma and breast cancer are related."
    doc = Document.create_simple_document(text)
    bc_ent = Entity.load_contiguous_entity(
        start=0,
        end=len(text),
        match="breast cancer",
        entity_class=parser.entity_class,
        namespace="test",
    )
    bc_ent.add_or_update_linking_candidates(candidates_with_metrics)
    doc.sections[0].entities.append(bc_ent)

    strategy = PreferNearestEmbeddingToDefaultLabelDisambiguationStrategy(
        confidence=DisambiguationConfidence.HIGHLY_LIKELY,
        complex_string_scorer=instantiate(kazu_test_config.SapbertStringSimilarityScorer),
    )

    strategy.prepare(doc)
    all_id_sets: set[EquivalentIdSet] = set(
        chain.from_iterable(candidate.associated_id_sets for candidate in bc_ent.linking_candidates)
    )

    disambiguated_id_sets = list(
        strategy.disambiguate(
            id_sets=all_id_sets,
            document=doc,
            parser_name=parser.name,
            ent_match=bc_ent.match,
            ent_match_norm=bc_ent.match_norm,
        )
    )
    assert len(disambiguated_id_sets) == 1
    assert disambiguated_id_sets[0].ids == {"1"}
