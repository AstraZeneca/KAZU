import copy
from typing import List, Dict
from kazu.data.data import Document, Entity
from kazu.steps.linking.entity_class_disambiguation import (
    EntityClassDisambiguationStep,
    DisambiguationEntry,
)


def test_EntityClassDisambiguationStep():
    context: Dict[str, List[DisambiguationEntry]] = {
        "UCB": [
            DisambiguationEntry(
                entity_class="anatomy", relevant_text=["umbilical pregnancy"], thresh=0.8
            ),
            DisambiguationEntry(
                entity_class="company", relevant_text=["pharma company brussels"], thresh=0.5
            ),
        ]
    }
    sents = [
        "Sentence context.",
        "UCB probably refers to the pharma company not umbilical cord blood.",
        "More sentence context.",
    ]
    doc = Document.simple_document_from_sents(sents)
    ucb_company_ent = Entity.load_contiguous_entity(
        start=len(sents[0]) + 1,
        end=len(sents[0]) + len("UCB") + 1,
        match="UCB",
        entity_class="company",
        namespace="test",
    )
    ucb_anatomy_ent = Entity.load_contiguous_entity(
        start=len(sents[0]) + 1,
        end=len(sents[0]) + len("UCB") + 1,
        match="UCB",
        entity_class="anatomy",
        namespace="test",
    )

    doc.sections[0].entities.extend([ucb_anatomy_ent, ucb_company_ent])

    step = EntityClassDisambiguationStep(context=context)
    step([doc])
    assert ucb_company_ent in doc.get_entities()
    assert ucb_anatomy_ent not in doc.get_entities()


def test_same_entity_with_same_span_in_multiple_sections():
    context: Dict[str, List[DisambiguationEntry]] = {
        "UCB": [
            DisambiguationEntry(
                entity_class="anatomy", relevant_text=["umbilical pregnancy"], thresh=0.8
            ),
            DisambiguationEntry(
                entity_class="company", relevant_text=["pharma company brussels"], thresh=0.5
            ),
        ]
    }
    section_a_sents = [
        "Sentence context.",
        "UCB probably refers to the pharma company not umbilical cord blood.",
        "More sentence context.",
    ]
    section_b_sents = [
        "Sentence context.",
        "UCB in this section refers to umbilical cord blood.",
        "More sentence context mentioning the word pregnancy."
    ]
    doc = Document.from_named_section_texts({"section_a": " ".join(section_a_sents),
                                             "section_b": " ".join(section_b_sents)})

    ucb_company_ent_section_a = Entity.load_contiguous_entity(
        start=len(section_a_sents[0]) + 1,
        end=len(section_a_sents[0]) + len("UCB") + 1,
        match="UCB",
        entity_class="company",
        namespace="test",
    )
    ucb_anatomy_ent_section_a = Entity.load_contiguous_entity(
        start=len(section_a_sents[0]) + 1,
        end=len(section_a_sents[0]) + len("UCB") + 1,
        match="UCB",
        entity_class="anatomy",
        namespace="test",
    )

    ucb_company_ent_section_b = copy.deepcopy(ucb_company_ent_section_a)
    ucb_anatomy_ent_section_b = copy.deepcopy(ucb_anatomy_ent_section_a)

    doc.sections[0].entities.extend([ucb_company_ent_section_a, ucb_anatomy_ent_section_a])
    doc.sections[1].entities.extend([ucb_company_ent_section_b, ucb_anatomy_ent_section_b])

    step = EntityClassDisambiguationStep(context=context)
    step([doc])

    # section_a should only have the UCB company entity
    assert ucb_company_ent_section_a in doc.get_entities()
    assert ucb_anatomy_ent_section_a not in doc.get_entities()

    # section_b should only have the UCB anatomy entity
    assert ucb_anatomy_ent_section_b in doc.get_entities()
    assert ucb_company_ent_section_b not in doc.get_entities()




