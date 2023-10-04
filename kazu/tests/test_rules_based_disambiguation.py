import pytest
from kazu.data.data import Document, Entity, Section
from kazu.steps.linking.rules_based_disambiguation import (
    RulesBasedEntityClassDisambiguationFilterStep,
)
from kazu.tests.utils import DummyParser

DRUG_TP_CLASS_BLOCK = [
    [
        {"_": {"drug": True}},
        {"LOWER": "is"},
        {"LOWER": "a"},
        {"LOWER": "molecule"},
    ]
]
DRUG_FP_CLASS_BLOCK = [
    [
        {"_": {"gene": True}},
        {"LOWER": "is"},
        {"LOWER": "a"},
        {"LOWER": "gene"},
    ]
]

GENE_TP_CLASS_BLOCK = [
    [
        {"_": {"gene": True}},
        {"LOWER": "is"},
        {"LOWER": "a"},
        {"LOWER": "gene"},
    ]
]
GENE_FP_CLASS_BLOCK = [
    [
        {"_": {"drug": True}},
        {"LOWER": "is"},
        {"LOWER": "a"},
        {"LOWER": "molecule"},
    ]
]
DRUG_TP_MENTION_BLOCK = [[{"LOWER": "drug"}]]
DRUG_FP_MENTION_BLOCK = [[{"LOWER": "protein"}]]
GENE_TP_MENTION_BLOCK = [[{"LOWER": "protein"}]]
GENE_FP_MENTION_BLOCK = [[{"LOWER": "drug"}]]
LOW_INFO_TEXT = "Insulin is commonly studied"


@pytest.mark.parametrize(
    ("class_matcher_rules", "mention_matcher_rules"),
    [
        pytest.param(
            {
                "drug": {
                    "tp": DRUG_TP_CLASS_BLOCK,
                    "fp": DRUG_FP_CLASS_BLOCK,
                },
                "gene": {
                    "tp": GENE_TP_CLASS_BLOCK,
                    "fp": GENE_FP_CLASS_BLOCK,
                },
            },
            {},
            id="all_tp_and_fp_class_matcher_rules",
        ),
        pytest.param(
            {
                "drug": {
                    "tp": DRUG_TP_CLASS_BLOCK,
                },
                "gene": {
                    "tp": GENE_TP_CLASS_BLOCK,
                },
            },
            {},
            id="all_tp_class_matcher_rules",
        ),
        pytest.param(
            {
                "drug": {
                    "fp": DRUG_FP_CLASS_BLOCK,
                },
                "gene": {
                    "fp": GENE_FP_CLASS_BLOCK,
                },
            },
            {},
            id="all_fp_class_matcher_rules",
        ),
        pytest.param(
            {
                "drug": {
                    "tp": DRUG_TP_CLASS_BLOCK,
                },
                "gene": {
                    "fp": GENE_FP_CLASS_BLOCK,
                },
            },
            {},
            id="single_tp_fp_class_matcher_rules",
        ),
        pytest.param(
            {},
            {
                "drug": {"Insulin": {"tp": DRUG_TP_MENTION_BLOCK, "fp": DRUG_FP_MENTION_BLOCK}},
                "gene": {"Insulin": {"tp": GENE_TP_MENTION_BLOCK, "fp": GENE_FP_MENTION_BLOCK}},
            },
            id="all_mention_rules",
        ),
        pytest.param(
            {
                "drug": {
                    "tp": DRUG_TP_CLASS_BLOCK,
                    "fp": DRUG_FP_CLASS_BLOCK,
                },
                "gene": {
                    "tp": GENE_TP_CLASS_BLOCK,
                    "fp": GENE_FP_CLASS_BLOCK,
                },
            },
            {
                "drug": {"Insulin": {"tp": DRUG_TP_MENTION_BLOCK, "fp": DRUG_FP_MENTION_BLOCK}},
                "gene": {"Insulin": {"tp": GENE_TP_MENTION_BLOCK, "fp": GENE_FP_MENTION_BLOCK}},
            },
            id="all_class_and_mention_rules",
        ),
    ],
)
def test_RulesBasedEntityClassDisambiguationFilterStep(
    class_matcher_rules, mention_matcher_rules, mock_kazu_disk_cache_on_parsers
):

    drug_text = "Insulin is a molecule or drug."
    gene_text = "Insulin is a gene or protein."

    drug_doc, gene_doc = _create_test_docs(drug_text, gene_text)
    parsers = [
        DummyParser(entity_class="drug", name="test1"),
        DummyParser(entity_class="gene", name="test2"),
    ]

    step = RulesBasedEntityClassDisambiguationFilterStep(
        class_matcher_rules=class_matcher_rules,
        mention_matcher_rules=mention_matcher_rules,
        parsers=parsers,
    )
    step([drug_doc, gene_doc])
    drug_ents = drug_doc.get_entities()
    assert len(drug_ents) == 2
    assert all(ent.entity_class == "drug" for ent in drug_ents)
    gene_ents = gene_doc.get_entities()
    assert len(gene_ents) == 2
    assert all(ent.entity_class == "gene" for ent in gene_ents)


def _create_test_docs(doc1_text: str, doc2_text: str):
    doc1 = Document.create_simple_document(doc1_text)
    doc1.sections.append(Section(name="test2", text=LOW_INFO_TEXT))
    doc1.sections[0].entities.append(
        Entity.load_contiguous_entity(
            start=0, end=7, entity_class="gene", namespace="test", match="Insulin"
        )
    )
    doc1.sections[0].entities.append(
        Entity.load_contiguous_entity(
            start=0, end=7, entity_class="drug", namespace="test", match="Insulin"
        )
    )
    doc1.sections[1].entities.append(
        Entity.load_contiguous_entity(
            start=0, end=7, entity_class="gene", namespace="test", match="Insulin"
        )
    )
    doc1.sections[1].entities.append(
        Entity.load_contiguous_entity(
            start=0, end=7, entity_class="drug", namespace="test", match="Insulin"
        )
    )
    doc2 = Document.create_simple_document(doc2_text)
    doc2.sections.append(Section(name="test2", text=LOW_INFO_TEXT))
    doc2.sections[0].entities.append(
        Entity.load_contiguous_entity(
            start=0, end=7, entity_class="gene", namespace="test", match="Insulin"
        )
    )
    doc2.sections[0].entities.append(
        Entity.load_contiguous_entity(
            start=0, end=7, entity_class="drug", namespace="test", match="Insulin"
        )
    )
    doc2.sections[1].entities.append(
        Entity.load_contiguous_entity(
            start=0, end=7, entity_class="gene", namespace="test", match="Insulin"
        )
    )
    doc2.sections[1].entities.append(
        Entity.load_contiguous_entity(
            start=0, end=7, entity_class="drug", namespace="test", match="Insulin"
        )
    )
    return doc1, doc2


def test_RulesBasedEntityClassDisambiguationFilterStep_pathological():
    patho_1 = "Insulin is a molecule or protein."  # fails on mention result
    patho_2 = "Insulin is a molecule or gene."  # fails on class result

    patho_1_doc, patho_2_doc = _create_test_docs(patho_1, patho_2)
    parsers = [
        DummyParser(entity_class="drug", name="test1"),
        DummyParser(entity_class="gene", name="test2"),
    ]
    step = RulesBasedEntityClassDisambiguationFilterStep(
        class_matcher_rules={
            "drug": {
                "tp": DRUG_TP_CLASS_BLOCK,  # type:ignore[dict-item]
                "fp": DRUG_FP_CLASS_BLOCK,  # type:ignore[dict-item]
            },
            "gene": {
                "tp": GENE_TP_CLASS_BLOCK,  # type:ignore[dict-item]
                "fp": GENE_FP_CLASS_BLOCK,  # type:ignore[dict-item]
            },
        },
        mention_matcher_rules={
            "drug": {"Insulin": {"tp": DRUG_TP_MENTION_BLOCK, "fp": DRUG_FP_MENTION_BLOCK}},
            "gene": {"Insulin": {"tp": GENE_TP_MENTION_BLOCK, "fp": GENE_FP_MENTION_BLOCK}},
        },
        parsers=parsers,
    )
    docs = [patho_1_doc, patho_2_doc]
    step(docs)
    for doc in docs:
        assert len(doc.get_entities()) == 0
