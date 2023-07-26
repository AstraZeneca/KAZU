from hydra.utils import instantiate

from kazu.data.data import Document, Entity
from kazu.tests.utils import requires_model_pack

test_text = (
    "BREXPIPRAZOLE is great and is the same as OPC-34712 but not 2,2'-ethylenedipyridine"
)


def check_step_has_found_entitites(doc, step_entity_class):
    for ent in doc.get_entities():
        if ent.entity_class == step_entity_class:
            for mapping in ent.mappings:
                assert mapping.source == "Opsin" and mapping.idx == "c1ccc(CCc2ccccn2)nc1"


@requires_model_pack
def test_opsin_step_no_condition(override_kazu_test_config):

    cfg = override_kazu_test_config(
        overrides=["~OpsinStep.condition"],
    )

    step = instantiate(cfg.OpsinStep)
    doc = Document.create_simple_document(test_text)
    processed_docs, failed_docs = step([doc])
    assert len(failed_docs) == 0
    check_step_has_found_entitites(processed_docs[0], step.entity_class)


@requires_model_pack
def test_opsin_step_with_condition(kazu_test_config):
    step = instantiate(kazu_test_config.OpsinStep)
    assert step.condition
    doc = Document.create_simple_document(test_text)
    processed_docs, failed_docs = step([doc])
    assert len(failed_docs) == 0
    assert all((x.entity_class != step.entity_class for x in doc.get_entities()))
    doc = Document.create_simple_document(test_text)
    doc.sections[0].entities.append(
        Entity.from_spans(
            text=test_text,
            spans=[(0, 13)],
            namespace="test",
            entity_class=next(iter(step.condition.required_entities)),
        )
    )
    doc.sections[0].entities.append(
        Entity.from_spans(
            text=test_text,
            spans=[(60, 83)],
            namespace="test",
            entity_class=next(iter(step.condition.required_entities)),
        )
    )
    processed_docs, failed_docs = step([doc])
    check_step_has_found_entitites(processed_docs[0], step.entity_class)
