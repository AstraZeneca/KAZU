import dataclasses

from hydra.utils import instantiate
from kazu.data import (
    Document,
    Entity,
    Section,
    Mapping,
    MentionConfidence,
    StringMatchConfidence,
    DisambiguationConfidence,
)
from kazu.steps.joint_ner_and_linking.explosion import ExplosionStringMatchingStep
from kazu.steps.other.cleanup import (
    CleanupStep,
    StripMappingURIsAction,
    DropMappingsByParserNameRankAction,
    DropEntityIfClassNotMatchedFilter,
    DropByMinLenFilter,
    DropEntityIfMatchInSetFilter,
)
from kazu.utils.utils import Singleton

doc_text = "XYZ1 is picked up as entity by explosion step but not mapped to a kb."
"ABC9 is picked up by a different NER step and also not mapped."
"But EGFR was picked up by explosion step and mapped."


def test_configured_mapping_cleanup_discards_ambiguous_mappings(kazu_test_config):
    action = instantiate(kazu_test_config.CleanupActions.MappingFilterCleanupAction)
    doc = Document.create_simple_document(doc_text)
    ents = [
        Entity.load_contiguous_entity(
            start=135,
            end=139,
            match="EGFR",
            entity_class="gene",
            namespace="test",
            mappings={
                Mapping(
                    default_label="EGFR",
                    source="test",
                    parser_name="test",
                    idx="test",
                    string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                    disambiguation_confidence=DisambiguationConfidence.HIGHLY_LIKELY,
                    string_match_strategy="test",
                    disambiguation_strategy=None,
                ),
                Mapping(
                    default_label="EGFR",
                    source="test",
                    parser_name="test",
                    idx="test",
                    string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                    disambiguation_confidence=DisambiguationConfidence.AMBIGUOUS,
                    string_match_strategy="test",
                    disambiguation_strategy=None,
                ),
            },
        ),
    ]

    doc.sections[0].entities.extend(ents)
    assert len(doc.get_entities()) == 1
    action.cleanup(doc)
    ent = doc.get_entities()[0]
    assert len(ent.mappings) == 1
    mapping = next(iter(ent.mappings))
    assert mapping.string_match_confidence == StringMatchConfidence.HIGHLY_LIKELY


def test_configured_entity_cleanup_discards_unmapped_explosion_ents(override_kazu_test_config):
    explosion_step_namespace = ExplosionStringMatchingStep.namespace()
    kazu_test_config = override_kazu_test_config(
        overrides=[
            f"CleanupActions.EntityFilterCleanupAction.filter_fns.0.from_ent_namespaces=[{explosion_step_namespace}]"
        ]
    )
    mock_other_ner_namespace = "mock_other_ner_namespace"

    action = instantiate(kazu_test_config.CleanupActions.EntityFilterCleanupAction)
    doc = Document.create_simple_document(doc_text)
    ents = [
        Entity.load_contiguous_entity(  # will get filtered out, no mappings and only POSSIBLE
            start=0,
            end=4,
            match="XYZ1",
            entity_class="gene",
            namespace=explosion_step_namespace,
            mention_confidence=MentionConfidence.POSSIBLE,
        ),
        Entity.load_contiguous_entity(  # will not get filtered out as it's higher confidence
            start=0,
            end=4,
            match="XYZ1",
            entity_class="gene",
            namespace=explosion_step_namespace,
            mention_confidence=MentionConfidence.PROBABLE,
        ),
        Entity.load_contiguous_entity(  # not filtered out - doesn't match the namespaces we're filtering
            start=69,
            end=73,
            match="ABC9",
            entity_class="gene",
            namespace=mock_other_ner_namespace,
            mention_confidence=MentionConfidence.POSSIBLE,
        ),
        Entity.load_contiguous_entity(  # not filtered out - has a mapping
            start=135,
            end=139,
            match="EGFR",
            entity_class="gene",
            namespace=explosion_step_namespace,
            mention_confidence=MentionConfidence.POSSIBLE,
            mappings={
                Mapping(
                    default_label="EGFR",
                    source="test",
                    parser_name="test",
                    idx="test",
                    string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                    string_match_strategy="test",
                    disambiguation_strategy=None,
                )
            },
        ),
    ]

    doc.sections[0].entities.extend(ents)
    assert len(doc.get_entities()) == 4
    action.cleanup(doc)
    assert len(doc.get_entities()) == 3


def test_uri_stripping_all_parsers(override_kazu_test_config):
    uri_stripping_conf = override_kazu_test_config(
        overrides=["CleanupActions=[default,uri_stripping]"]
    )
    action = instantiate(uri_stripping_conf.CleanupActions.StripMappingURIsAction)
    doc = Document.create_simple_document(doc_text)
    short_id_egfr_mapping = Mapping(
        default_label="EGFR",
        source="ENSEMBL",
        parser_name="opentargets_target",
        idx="OPENTARGETS_TARGET",  # short id, should remain the same
        string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
        disambiguation_confidence=DisambiguationConfidence.HIGHLY_LIKELY,
        string_match_strategy="test",
        disambiguation_strategy=None,
    )
    long_id_egfr_mapping = Mapping(
        default_label="EGFR",
        source="ENSEMBL",
        parser_name="opentargets_target_but_long_ids",
        idx="http://identifiers.org/Ensembl:ENSG00000146648",  # should get stripped
        # note that this ID isn't used in general by kazu, but is used by OXO
        # for EGFR: https://www.ebi.ac.uk/spot/oxo/terms/ENSG00000146648
        # we have URIs for entities in a number of KBs, such as MONDO
        # (generally anything parsed with RDFGraphParser or a subclass)
        string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
        disambiguation_confidence=DisambiguationConfidence.AMBIGUOUS,
        string_match_strategy="test",
        disambiguation_strategy=None,
    )

    ents = [
        Entity.load_contiguous_entity(
            start=135,
            end=139,
            match="EGFR",
            entity_class="gene",
            namespace="test",
            mappings={short_id_egfr_mapping, long_id_egfr_mapping},
        ),
    ]
    doc.sections[0].entities.extend(ents)

    assert len(doc.get_entities()) == 1
    action.cleanup(doc)
    ent = doc.get_entities()[0]
    assert ent.mappings == {
        short_id_egfr_mapping,
        dataclasses.replace(long_id_egfr_mapping, idx="Ensembl:ENSG00000146648"),
    }


def test_uri_stripping_only_some_parsers():
    action = StripMappingURIsAction(parsers_to_strip=["mondo", "clo"])
    (
        asthma_mapping_mondo,
        asthma_mapping_other_ont,
        doc,
        hsc0054_mapping_clo,
        hsc0054_mapping_other_ont,
    ) = set_up_simple_cleanup_test_case()

    assert len(doc.get_entities()) == 2
    action.cleanup(doc)
    asthma_ent = doc.get_entities()[0]
    assert asthma_ent.mappings == {
        dataclasses.replace(asthma_mapping_mondo, idx="MONDO_0004979"),
        asthma_mapping_other_ont,
    }
    hsc0054_ent = doc.get_entities()[1]
    assert hsc0054_ent.mappings == {
        dataclasses.replace(hsc0054_mapping_clo, idx="CLO_0051085"),
        hsc0054_mapping_other_ont,
    }


def set_up_simple_cleanup_test_case():
    doc = Document.create_simple_document("Asthma is in mondo and HSC0054 is a cell line in CLO.")
    asthma_mapping_mondo = Mapping(
        default_label="Asthma",
        source="MONDO",
        parser_name="mondo",
        idx="http://purl.obolibrary.org/obo/MONDO_0004979",
        string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
        disambiguation_confidence=DisambiguationConfidence.HIGHLY_LIKELY,
        string_match_strategy="test",
        disambiguation_strategy=None,
    )
    asthma_mapping_other_ont = dataclasses.replace(asthma_mapping_mondo, parser_name="not_mondo")
    hsc0054_mapping_clo = Mapping(
        default_label="HSC0054",
        source="CLO",
        parser_name="clo",
        idx="http://purl.obolibrary.org/obo/CLO_0051085",
        string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
        disambiguation_confidence=DisambiguationConfidence.HIGHLY_LIKELY,
        string_match_strategy="test",
        disambiguation_strategy=None,
    )
    hsc0054_mapping_other_ont = dataclasses.replace(hsc0054_mapping_clo, parser_name="not_clo")
    ents = [
        Entity.load_contiguous_entity(
            start=0,
            end=6,
            match="Asthma",
            entity_class="disease",
            namespace="test",
            mappings={asthma_mapping_mondo, asthma_mapping_other_ont},
        ),
        Entity.load_contiguous_entity(
            start=23,
            end=30,
            match="HSC0054",
            entity_class="cell_line",
            namespace="test",
            mappings={hsc0054_mapping_clo, hsc0054_mapping_other_ont},
        ),
    ]
    doc.sections[0].entities.extend(ents)
    return (
        asthma_mapping_mondo,
        asthma_mapping_other_ont,
        doc,
        hsc0054_mapping_clo,
        hsc0054_mapping_other_ont,
    )


def test_drop_by_parser_name_rank():
    (
        asthma_mapping_mondo,
        asthma_mapping_other_ont,
        doc,
        hsc0054_mapping_clo,
        hsc0054_mapping_other_ont,
    ) = set_up_simple_cleanup_test_case()

    Singleton.clear_all()

    action = DropMappingsByParserNameRankAction(
        entity_class_to_parser_name_rank={
            "disease": [asthma_mapping_mondo.parser_name, asthma_mapping_other_ont.parser_name],
            "cell_line": [hsc0054_mapping_other_ont.parser_name, hsc0054_mapping_clo.parser_name],
        }
    )
    action.cleanup(doc)
    for entity in doc.get_entities():
        if entity.entity_class == "disease":
            assert entity.mappings == {asthma_mapping_mondo}
        else:
            assert entity.mappings == {hsc0054_mapping_other_ont}


def test_drop_by_min_len():
    filter_func = DropByMinLenFilter(min_len=2)
    long_ent = Entity.load_contiguous_entity(
        start=0,
        end=2,
        match="lo",
        entity_class="test",
        namespace="test",
        mappings=None,
    )
    assert not filter_func(long_ent)
    short_ent = Entity.load_contiguous_entity(
        start=0,
        end=1,
        match="l",
        entity_class="test",
        namespace="test",
        mappings=None,
    )
    assert filter_func(short_ent)


def test_drop_entity_if_class_not_matched():
    filter_func = DropEntityIfClassNotMatchedFilter(required_classes=["required1", "required2"])
    required_ents = [
        Entity.load_contiguous_entity(
            start=0,
            end=2,
            match="lo",
            entity_class="required1",
            namespace="test",
            mappings=None,
        ),
        Entity.load_contiguous_entity(
            start=0,
            end=2,
            match="lo",
            entity_class="required2",
            namespace="test",
            mappings=None,
        ),
    ]
    assert not any(filter_func(rec) for rec in required_ents)
    discard_ents = [
        Entity.load_contiguous_entity(
            start=0,
            end=2,
            match="lo",
            entity_class="discard1",
            namespace="test",
            mappings=None,
        ),
        Entity.load_contiguous_entity(
            start=0,
            end=2,
            match="lo",
            entity_class="discard2",
            namespace="test",
            mappings=None,
        ),
    ]
    assert all(filter_func(rec) for rec in discard_ents)


def test_drop_ent_if_match_in_set_filter():
    filter_func = DropEntityIfMatchInSetFilter(
        {"gene": ["abd", "def"], "disease": ["disease1", "disease1"]}
    )
    required_ents = [
        Entity.load_contiguous_entity(
            start=0,
            end=2,
            match="required",
            entity_class="gene",
            namespace="test",
            mappings=None,
        ),
        Entity.load_contiguous_entity(
            start=0,
            end=2,
            match="required2",
            entity_class="disease",
            namespace="test",
            mappings=None,
        ),
        Entity.load_contiguous_entity(
            start=0,
            end=2,
            match="required3",
            entity_class="drug",
            namespace="test",
            mappings=None,
        ),
    ]
    assert not any(filter_func(rec) for rec in required_ents)
    discard_ents = [
        Entity.load_contiguous_entity(
            start=0,
            end=2,
            match="abd",
            entity_class="gene",
            namespace="test",
            mappings=None,
        ),
        Entity.load_contiguous_entity(
            start=0,
            end=2,
            match="disease1",
            entity_class="disease",
            namespace="test",
            mappings=None,
        ),
    ]
    assert all(filter_func(rec) for rec in discard_ents)


def test_cleanup_step():
    class MockCleanupAction1:
        def cleanup(self, doc: Document):
            doc.sections = [section for section in doc.sections if len(section.text) >= 3]

    class MockCleanupAction2:
        def cleanup(self, doc: Document):
            for ent in doc.get_entities():
                if ent.namespace == "tricky_ent_step":
                    raise Exception(f"{self.__class__} fails on ents from {ent.namespace}!")
                else:
                    ent.match = ent.match.upper()

    cleanup_step = CleanupStep(cleanup_actions=[MockCleanupAction1(), MockCleanupAction2()])
    doc1 = Document(
        idx="test1",
        sections=[
            Section(text="hi", name="doc1_section1"),
            Section(text="2nd section in doc1", name="doc1_section2"),
        ],
    )
    doc2 = Document.create_simple_document(text="cursed document with a gremlin entity")
    doc2.sections[0].entities.append(
        Entity.load_contiguous_entity(
            start=23,
            end=30,
            match="gremlin",
            entity_class="tricky_entity",
            namespace="tricky_ent_step",
        )
    )

    assert len(doc1.sections) == 2
    docs, failed_docs = cleanup_step([doc1, doc2])
    assert len(docs) == 2
    assert len(failed_docs) == 1
    assert len(doc1.sections) == 1
