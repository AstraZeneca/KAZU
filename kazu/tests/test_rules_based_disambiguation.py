import pytest
from kazu.data.data import Document, Entity, Section
from kazu.steps.linking.rule_based_disambiguator import (
    RulesBasedEntityClassDisambiguationFilterStep,
)


@pytest.mark.parametrize(
    ("class_matcher_rules", "cooccurence_rules"),
    [
        pytest.param(
            {
                "drug": {
                    "tp": [
                        [
                            {"_": {"drug": True}},
                            {"LOWER": "is"},
                            {"LOWER": "a"},
                            {"LOWER": "molecule"},
                        ]
                    ],
                    "fp": [
                        [
                            {"_": {"gene": True}},
                            {"LOWER": "is"},
                            {"LOWER": "a"},
                            {"LOWER": "gene"},
                        ]
                    ],
                },
                "gene": {
                    "tp": [
                        [
                            {"_": {"gene": True}},
                            {"LOWER": "is"},
                            {"LOWER": "a"},
                            {"LOWER": "gene"},
                        ]
                    ],
                    "fp": [
                        [
                            {"_": {"drug": True}},
                            {"LOWER": "is"},
                            {"LOWER": "a"},
                            {"LOWER": "molecule"},
                        ]
                    ],
                },
            },
            {},
            id="class_matcher_rules",
        ),
        pytest.param(
            {},
            {
                "drug": {"Insulin": {"tp": ["molecule"], "fp": ["gene"]}},
                "gene": {"Insulin": {"fp": ["molecule"], "tp": ["gene"]}},
            },
            id="cooccurrence_rules",
        ),
    ],
)
def test_RulesBasedEntityClassDisambiguationFilterStep(class_matcher_rules, cooccurence_rules):

    drug_text = "Insulin is a molecule."
    gene_text = "Insulin is a gene."
    low_info_text = "Insulin is commonly studied"

    drug_doc = Document.create_simple_document(drug_text)
    drug_doc.sections.append(Section(name="test2", text=low_info_text))
    drug_doc.sections[0].entities.append(
        Entity.load_contiguous_entity(
            start=0, end=7, entity_class="gene", namespace="test", match="Insulin"
        )
    )
    drug_doc.sections[0].entities.append(
        Entity.load_contiguous_entity(
            start=0, end=7, entity_class="drug", namespace="test", match="Insulin"
        )
    )
    drug_doc.sections[1].entities.append(
        Entity.load_contiguous_entity(
            start=0, end=7, entity_class="gene", namespace="test", match="Insulin"
        )
    )
    drug_doc.sections[1].entities.append(
        Entity.load_contiguous_entity(
            start=0, end=7, entity_class="drug", namespace="test", match="Insulin"
        )
    )

    gene_doc = Document.create_simple_document(gene_text)
    gene_doc.sections.append(Section(name="test2", text=low_info_text))
    gene_doc.sections[0].entities.append(
        Entity.load_contiguous_entity(
            start=0, end=7, entity_class="gene", namespace="test", match="Insulin"
        )
    )
    gene_doc.sections[0].entities.append(
        Entity.load_contiguous_entity(
            start=0, end=7, entity_class="drug", namespace="test", match="Insulin"
        )
    )
    gene_doc.sections[1].entities.append(
        Entity.load_contiguous_entity(
            start=0, end=7, entity_class="gene", namespace="test", match="Insulin"
        )
    )
    gene_doc.sections[1].entities.append(
        Entity.load_contiguous_entity(
            start=0, end=7, entity_class="drug", namespace="test", match="Insulin"
        )
    )

    step = RulesBasedEntityClassDisambiguationFilterStep(
        class_matcher_rules=class_matcher_rules, cooccurrence_rules=cooccurence_rules
    )
    step([drug_doc, gene_doc])
    drug_ents = drug_doc.get_entities()
    assert len(drug_ents) == 2
    assert all(ent.entity_class == "drug" for ent in drug_ents)
    gene_ents = gene_doc.get_entities()
    assert len(gene_ents) == 2
    assert all(ent.entity_class == "gene" for ent in gene_ents)
