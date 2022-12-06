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
        start=len(sents[0]),
        end=len(sents[0]) + len("UCB"),
        match="UCB",
        entity_class="company",
        namespace="test",
    )
    ucb_anatomy_ent = Entity.load_contiguous_entity(
        start=len(sents[0]),
        end=len(sents[0]) + len("UCB"),
        match="UCB",
        entity_class="anatomy",
        namespace="test",
    )

    doc.sections[0].entities.extend([ucb_anatomy_ent, ucb_company_ent])

    step = EntityClassDisambiguationStep(context=context)
    step([doc])
    assert ucb_company_ent in doc.get_entities()
    assert ucb_anatomy_ent not in doc.get_entities()
