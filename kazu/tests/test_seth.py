from hydra.utils import instantiate

from kazu.data.data import Document, Entity
from kazu.tests.utils import requires_model_pack

test_text = (
    "Causative GJB2 mutations were identified in 31 (15.2%) patients, and two common mutations, c.35delG and "
    "L90P (c.269T>C), accounted for 72.1% and 9.8% of GJB2 disease alleles."
)

unfound_correct_hgvs_strings = {"p.Leu90Pro", "c.269T>C", "c.35delG"}


def check_step_has_found_entitites(doc, step_entity_class):
    for ent in doc.get_entities():
        if ent.entity_class == step_entity_class:
            unfound_correct_hgvs_strings.discard(ent.metadata["hgvs"])
    assert not unfound_correct_hgvs_strings


@requires_model_pack
def test_seth_step_no_condition(override_kazu_test_config):

    cfg = override_kazu_test_config(
        overrides=["~SethStep.condition"],
    )

    step = instantiate(cfg.SethStep)
    doc = Document.create_simple_document(test_text)
    success_docs, failed_docs = step([doc])
    assert len(failed_docs) == 0
    check_step_has_found_entitites(success_docs[0], step.entity_class)


def test_seth_step_with_condition(kazu_test_config):
    step = instantiate(kazu_test_config.SethStep)
    assert step.condition
    doc = Document.create_simple_document(test_text)
    success_docs, failed_docs = step([doc])
    assert len(failed_docs) == 0
    assert all((x.entity_class != step.entity_class for x in doc.get_entities()))
    doc = Document.create_simple_document(test_text)
    doc.sections[0].entities.append(
        Entity.from_spans(
            text=test_text,
            spans=[(0, 5)],
            namespace="test",
            entity_class=next(iter(step.condition.required_entities)),
        )
    )
    success_docs, failed_docs = step([doc])
    check_step_has_found_entitites(success_docs[0], step.entity_class)
