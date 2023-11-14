from spacy.tokens import Token
from spacy.tokens.underscore import Underscore
import pytest

from kazu.data.data import Section, Entity
from kazu.utils.spacy_object_mapper import SpacyToKazuObjectMapper


# see https://github.com/explosion/spaCy/discussions/5424#discussioncomment-189880
# code from https://github.com/explosion/spaCy/blob/f333c2a0114589121b10ef93bd92eb27b7d78d2e/spacy/tests/doc/test_underscore.py#L10-L16
@pytest.fixture(scope="function", autouse=True)
def clean_underscore():
    # reset the Underscore object after the test, to avoid having state copied across tests
    yield
    Underscore.doc_extensions = {}
    Underscore.span_extensions = {}
    Underscore.token_extensions = {}


drug_gene_and_disease = ["drug", "gene", "disease"]


sample_text = """Paracetamol is a drug.
EGFR is a gene.
NSCLC is a disease.
Hand is an anatomical entity.
AstraZeneca is a company.
"""
egfr_start = sample_text.find("EGFR")
nsclc_start = sample_text.find("NSCLC")
hand_start = sample_text.find("Hand")
az_start = sample_text.find("AstraZeneca")

paracetamol = Entity.load_contiguous_entity(
    match="Paracetamol", entity_class="drug", start=0, end=len("Paracetamol"), namespace="test"
)
egfr = Entity.load_contiguous_entity(
    match="EGFR",
    entity_class="gene",
    start=egfr_start,
    end=egfr_start + len("EGFR"),
    namespace="test",
)
nsclc = Entity.load_contiguous_entity(
    match="NSCLC",
    entity_class="disease",
    start=nsclc_start,
    end=nsclc_start + len("NSCLC"),
    namespace="test",
)
hand = Entity.load_contiguous_entity(
    match="Hand",
    entity_class="anatomy",
    start=hand_start,
    end=hand_start + len("Hand"),
    namespace="test",
)
az = Entity.load_contiguous_entity(
    match="AstraZeneca",
    entity_class="company",
    start=az_start,
    end=az_start + len("AstraZeneca"),
    namespace="test",
)
sample_section = Section(
    text=sample_text,
    name="sample",
    entities=[
        paracetamol,
        egfr,
        nsclc,
        hand,
        az,
    ],
)


@pytest.mark.parametrize(
    "entity_classes",
    [
        pytest.param({}, id="empty_entity_classes"),
        pytest.param(drug_gene_and_disease, id="part_populated_entity_classes"),
    ],
)
def test_object_mapper_incremental_attributes(entity_classes):
    mapper = SpacyToKazuObjectMapper(
        entity_classes=entity_classes, set_attributes_incrementally=True
    )

    mapped_result = mapper(sample_section)

    assert len(mapped_result) == len(sample_section.entities)

    for entity, span in mapped_result.items():
        for token in span:
            assert token._.get(entity.entity_class) is True

    # mapper should have all entity classes now (and class is unique to each entity)
    assert len(mapper.entity_classes) == len(sample_section.entities)


def test_object_mapper_non_incremental_attributes():
    mapper = SpacyToKazuObjectMapper(
        entity_classes=drug_gene_and_disease, set_attributes_incrementally=False
    )

    mapped_result = mapper(sample_section)

    assert len(mapped_result) == len(sample_section.entities)

    for entity_with_covered_class in (paracetamol, egfr, nsclc):
        for token in mapped_result[entity_with_covered_class]:
            assert token._.get(entity_with_covered_class.entity_class) is True

    for entity_with_uncovered_class in (hand, az):
        assert not Token.has_extension(entity_with_uncovered_class.entity_class)

    # mapper should still only have initial classes explicitly set
    assert len(mapper.entity_classes) == len(drug_gene_and_disease)
