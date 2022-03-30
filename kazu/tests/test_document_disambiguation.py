from hydra.utils import instantiate
from kazu.data.data import (
    Document,
    Entity,
    Mapping,
    LINK_UNCERTAINTY,
    LINK_CONFIDENCE,
    AMBIGUOUS_IDX,
    SynonymData,
)
from kazu.data.data import LinkRanks
from kazu.modelling.ontology_preprocessing.base import MetadataDatabase
from kazu.steps.other.document_disambiguation import (
    DISAMBIGUATED_BY,
    DISAMBIGUATED_BY_DEFINED_ELSEWHERE,
    DISAMBIGUATED_BY_REACTOME,
)
from kazu.tests.utils import requires_model_pack
from kazu.utils.link_index import Hit, DICTIONARY_HITS


@requires_model_pack
def test_kb_disambiguation(kazu_test_config):
    metadata_db = MetadataDatabase()
    EXPECTED_IDX = "ENSG00000111276"
    PATHWAY_LINKED_IDX = "ENSG00000110092"
    WRONG_IDX = "ENSG00000110801"
    metadata_db.add(
        "ENSEMBL", {EXPECTED_IDX: {"default_label": "cyclin dependent kinase inhibitor 1B"}}
    )

    step = instantiate(kazu_test_config.DocumentLevelDisambiguationStep)
    text = "p27 can be many things, but it often co-occurs with cyclin D1"
    doc = Document.create_simple_document(text)

    # Two entities, one ambiguous entities with one mapping, one unambiguous with one mapping. Both connected via
    # pathway data

    ambig_ent = Entity.load_contiguous_entity(
        start=0, end=3, match="p27", entity_class="test", namespace="test"
    )
    ambig_ent.mappings = [
        Mapping(
            default_label="wrong",
            source="ENSEMBL",
            idx=AMBIGUOUS_IDX,
            mapping_type=frozenset(),
            metadata={
                LINK_CONFIDENCE: LinkRanks.AMBIGUOUS,
                DICTIONARY_HITS: [
                    Hit(
                        matched_str="",
                        confidence=LinkRanks.MEDIUM_CONFIDENCE,
                        syn_data=frozenset(
                            [
                                SynonymData(ids=frozenset([WRONG_IDX])),
                                SynonymData(ids=frozenset([EXPECTED_IDX])),
                            ]
                        ),
                    )
                ],
            },
        )
    ]
    unambig_ent = Entity.load_contiguous_entity(
        start=len(text) - 9, end=len(text), match="cyclin D1", entity_class="test", namespace="test"
    )
    unambig_ent.mappings = [
        Mapping(
            default_label="right",
            source="ENSEMBL",
            idx=PATHWAY_LINKED_IDX,
            mapping_type=frozenset(),
            metadata={LINK_CONFIDENCE: LinkRanks.HIGH_CONFIDENCE},
        )
    ]
    doc.sections[0].entities = [ambig_ent, unambig_ent]
    step([doc])
    result_entities = doc.get_entities()
    assert len(result_entities) == 2
    assert len(result_entities[0].mappings) == 1
    assert len(result_entities[1].mappings) == 1
    assert result_entities[0].mappings[0].idx == EXPECTED_IDX
    assert result_entities[0].mappings[0].default_label == "cyclin dependent kinase inhibitor 1B"
    assert (
        result_entities[0].mappings[0].metadata.get(DISAMBIGUATED_BY) == DISAMBIGUATED_BY_REACTOME
    )
    assert result_entities[1].mappings[0].idx == "ENSG00000110092"


@requires_model_pack
def test_document_disambiguation_s1(kazu_test_config):
    # scenario 1. Two entities, one entity with one ambiguous mapping, one with one unambiguous mapping
    step = instantiate(kazu_test_config.DocumentLevelDisambiguationStep)
    EXPECTED_IDX = "I'm right"
    text = "p27 can be many things, but in this case it's CDKN1B"
    doc = Document.create_simple_document(text)

    ambig_ent = Entity.load_contiguous_entity(
        start=0, end=3, match="p27", entity_class="test", namespace="test"
    )
    ambig_ent.mappings = [
        Mapping(
            default_label="wrong",
            source="test_kb",
            idx=AMBIGUOUS_IDX,
            mapping_type=frozenset(),
            metadata={
                LINK_CONFIDENCE: LinkRanks.AMBIGUOUS,
                DICTIONARY_HITS: [
                    Hit(
                        matched_str="",
                        confidence=LinkRanks.MEDIUM_CONFIDENCE,
                        syn_data=frozenset(
                            [
                                SynonymData(ids=frozenset(["im wrong"])),
                                SynonymData(ids=frozenset([EXPECTED_IDX])),
                            ]
                        ),
                    )
                ],
            },
        )
    ]
    unambig_ent = Entity.load_contiguous_entity(
        start=len(text) - 6, end=len(text), match="CDKN1B", entity_class="test", namespace="test"
    )
    unambig_ent.mappings = [
        Mapping(
            default_label="right",
            source="test_kb",
            idx=EXPECTED_IDX,
            mapping_type=frozenset(),
            metadata={LINK_CONFIDENCE: LinkRanks.HIGH_CONFIDENCE},
        )
    ]
    doc.sections[0].entities = [ambig_ent, unambig_ent]
    step([doc])
    result_entities = doc.get_entities()
    assert len(result_entities) == 2
    assert len(result_entities[0].mappings) == 1
    assert len(result_entities[1].mappings) == 1
    assert result_entities[0].mappings[0].idx == EXPECTED_IDX
    assert result_entities[1].mappings[0].idx == EXPECTED_IDX


@requires_model_pack
def test_document_disambiguation_s2(kazu_test_config):
    # scenario 2. Three entities, two ambiguous entities with one mapping, one unambiguous with one mapping
    step = instantiate(kazu_test_config.DocumentLevelDisambiguationStep)
    EXPECTED_IDX = "I'm right"

    text = "p27 can be many things, but in this case it's CDKN1B"
    doc = Document.create_simple_document(text)

    ambig_ent = Entity.load_contiguous_entity(
        start=0, end=3, match="p27", entity_class="test", namespace="test"
    )
    ambig_ent.mappings = [
        Mapping(
            default_label="wrong id",
            source="test_kb",
            idx=AMBIGUOUS_IDX,
            mapping_type=frozenset(),
            metadata={
                LINK_CONFIDENCE: LinkRanks.AMBIGUOUS,
                DICTIONARY_HITS: [
                    Hit(
                        matched_str="",
                        confidence=LinkRanks.MEDIUM_CONFIDENCE,
                        syn_data=frozenset(
                            [
                                SynonymData(ids=frozenset(["im wrong"])),
                                SynonymData(ids=frozenset([EXPECTED_IDX])),
                            ]
                        ),
                    )
                ],
            },
        )
    ]
    unambig_ent_low_conf = Entity.load_contiguous_entity(
        start=0, end=3, match="p27", entity_class="test", namespace="test"
    )
    unambig_ent_low_conf.mappings = [
        Mapping(
            default_label="right id, low confidence",
            source="test_kb",
            idx=EXPECTED_IDX,
            mapping_type=frozenset(),
            metadata={LINK_CONFIDENCE: LinkRanks.LOW_CONFIDENCE},
        )
    ]

    unambig_ent = Entity.load_contiguous_entity(
        start=len(text) - 6, end=len(text), match="CDKN1B", entity_class="test", namespace="test"
    )
    unambig_ent.mappings = [
        Mapping(
            default_label="right",
            source="test_kb",
            idx=EXPECTED_IDX,
            mapping_type=frozenset(),
            metadata={LINK_CONFIDENCE: LinkRanks.HIGH_CONFIDENCE},
        )
    ]
    doc.sections[0].entities = [ambig_ent, ambig_ent, unambig_ent]
    step([doc])
    result_entities = doc.get_entities()
    assert len(result_entities) == 3
    assert len(result_entities[0].mappings) == 1
    assert len(result_entities[1].mappings) == 1
    assert len(result_entities[2].mappings) == 1
    assert result_entities[0].mappings[0].idx == EXPECTED_IDX
    assert result_entities[1].mappings[0].idx == EXPECTED_IDX
    assert result_entities[0].mappings[0].idx == EXPECTED_IDX
    assert (
        result_entities[0].mappings[0].metadata.get(DISAMBIGUATED_BY)
        == DISAMBIGUATED_BY_DEFINED_ELSEWHERE
    )
    assert (
        result_entities[1].mappings[0].metadata.get(DISAMBIGUATED_BY)
        == DISAMBIGUATED_BY_DEFINED_ELSEWHERE
    )


@requires_model_pack
def test_document_disambiguation_s3(kazu_test_config):
    # scenario 3. Two entities, one entity with one ambiguous mapping and one unabiguous mapping, one with one unambiguous mapping
    step = instantiate(kazu_test_config.DocumentLevelDisambiguationStep)
    EXPECTED_IDX = "I'm right"
    text = "p27 can be many things, but in this case it's CDKN1B"
    doc = Document.create_simple_document(text)

    # scenario 3. Two entities, one entity with one ambiguous mapping from kb X and one unambiguous mapping from kb Y
    # one with one unambiguous mapping. We should preserve the uncertain one, so it can be handled elsewhere
    ambig_ent = Entity.load_contiguous_entity(
        start=0, end=3, match="p27", entity_class="test", namespace="test"
    )
    ambig_ent.mappings = [
        Mapping(
            default_label="maybe right",
            source="kb_x",
            idx=AMBIGUOUS_IDX,
            mapping_type=frozenset(),
            metadata={
                LINK_CONFIDENCE: LinkRanks.AMBIGUOUS,
                DICTIONARY_HITS: [
                    Hit(
                        matched_str="",
                        confidence=LinkRanks.MEDIUM_CONFIDENCE,
                        syn_data=frozenset(
                            [
                                SynonymData(ids=frozenset(["I may be right"])),
                                SynonymData(ids=frozenset(["I also may be right"])),
                            ]
                        ),
                    )
                ],
            },
        ),
        Mapping(
            default_label="right",
            source="kb_y",
            idx=EXPECTED_IDX,
            mapping_type=frozenset(),
            metadata={LINK_CONFIDENCE: LinkRanks.HIGH_CONFIDENCE},
        ),
    ]
    unambig_ent = Entity.load_contiguous_entity(
        start=len(text) - 6, end=len(text), match="CDKN1B", entity_class="test", namespace="test"
    )
    unambig_ent.mappings = [
        Mapping(
            default_label="right",
            source="test_kb",
            idx=EXPECTED_IDX,
            mapping_type=frozenset(),
            metadata={LINK_CONFIDENCE: LinkRanks.HIGH_CONFIDENCE},
        )
    ]
    doc.sections[0].entities = [ambig_ent, unambig_ent]
    step([doc])
    result_entities = doc.get_entities()
    assert len(result_entities) == 2
    assert len(result_entities[0].mappings) == 2
    assert len(result_entities[1].mappings) == 1
    assert result_entities[0].mappings[1].idx == EXPECTED_IDX
    assert result_entities[0].mappings[0].idx == AMBIGUOUS_IDX
    assert result_entities[0].mappings[0].default_label == "maybe right"
    assert result_entities[1].mappings[0].idx == EXPECTED_IDX
