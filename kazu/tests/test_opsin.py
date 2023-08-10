from hydra.utils import instantiate

from kazu.data.data import Document, Entity
from kazu.tests.utils import requires_model_pack

test_text = "A test: BREXPIPRAZOLE is great and is the same as OPC-34712 but not Bicyclo[3.2.1]octane or 2,2'-ethylenedipyridine or Benzo[1\",2\":3,4;4\",5\":3',4']dicyclobuta[1,2-b:1',2'-c']difuran or Cyclohexanone ethyl methyl ketal or 4-[2-(2-chloro-4-fluoroanilino)-5-methylpyrimidin-4-yl]-N-[(1S)-1-(3-chlorophenyl)-2-hydroxyethyl]-1H-pyrrole-2-carboxamide added to 7-cyclopentyl-5-(4-methoxyphenyl)pyrrolo[2,3-d]pyrimidin-4-amine"
test_smiles = [
    "C1CC2CCC(C1)C2",
    "c1ccc(CCc2ccccn2)nc1",
    "c1cc2c3cc4c5cocc5c4cc3c2o1",
    "O=C1CCCCC1",
    "CCOC1(OC)CCCCC1",
    "Cc1cnc(Nc2ccc(F)cc2Cl)nc1-c1c[nH]c(C(=O)N[C@H](CO)c2cccc(Cl)c2)c1",
    "COc1ccc(-c2cn(C3CCCC3)c3ncnc(N)c23)cc1",
]


def check_step_has_found_entities(doc, step_entity_class) -> int:
    mappings = 0
    for ent in doc.get_entities():
        if ent.entity_class == step_entity_class:
            for mapping in ent.mappings:
                assert mapping.source == "Opsin" and mapping.idx in test_smiles
                mappings = mappings + 1
    return mappings


@requires_model_pack
def test_opsin_step_no_condition(override_kazu_test_config):

    cfg = override_kazu_test_config(
        overrides=["~OpsinStep.condition"],
    )

    step = instantiate(cfg.OpsinStep)
    doc = Document.create_simple_document(test_text)
    doc.sections[0].entities.append(
        Entity.from_spans(
            text=test_text,
            spans=[(68, 88)],
            namespace="test",
            entity_class=step.entity_class,
        )
    )
    processed_docs, failed_docs = step([doc])
    assert len(failed_docs) == 0
    assert check_step_has_found_entities(processed_docs[0], step.entity_class) == 0


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
            spans=[(68, 88)],  # Bicyclo[3.2.1]octane
            namespace="test",
            entity_class=next(iter(step.condition.required_entities)),
        )
    )
    doc.sections[0].entities.append(
        Entity.from_spans(
            text=test_text,
            spans=[(68, 88)],  # Bicyclo[3.2.1]octane
            namespace="test",
            entity_class="some irrelevant entity class",  # this should not be mapped because of invalid entity class
        )
    )
    doc.sections[0].entities.append(
        Entity.from_spans(
            text=test_text,
            spans=[(68, 82)],  # mimic transformer partial match 'Bicyclo[3.2.1]'
            namespace="test",
            entity_class=next(iter(step.condition.required_entities)),
        )
    )
    doc.sections[0].entities.append(
        Entity.from_spans(
            text=test_text,
            spans=[(92, 115)],  # 2,2'-ethylenedipyridine
            namespace="test",
            entity_class=next(iter(step.condition.required_entities)),
        )
    )
    doc.sections[0].entities.append(
        Entity.from_spans(
            text=test_text,
            spans=[
                (119, 181)
            ],  # Benzo[1\",2\":3,4;4\",5\":3',4']dicyclobuta[1,2-b:1',2'-c']difuran
            namespace="test",
            entity_class=next(iter(step.condition.required_entities)),
        )
    )
    doc.sections[0].entities.append(
        Entity.from_spans(
            text=test_text,
            spans=[(185, 198)],  # mimic transformer partial match 'Cyclohexanone'
            # won't expand to 'Cyclohexanone ethyl methyl ketal' because we only look two spaces out currently
            namespace="test",
            entity_class=next(iter(step.condition.required_entities)),
        )
    )
    doc.sections[0].entities.append(
        Entity.from_spans(
            text=test_text,
            spans=[(185, 217)],  # Cyclohexanone ethyl methyl ketal
            namespace="test",
            entity_class=next(iter(step.condition.required_entities)),
        )
    )
    doc.sections[0].entities.append(
        Entity.from_spans(
            text=test_text,
            spans=[(221, 334)],
            namespace="test",
            entity_class=next(iter(step.condition.required_entities)),
        )
    )
    doc.sections[0].entities.append(
        Entity.from_spans(
            text=test_text,
            spans=[(354, 418)],
            namespace="test",
            entity_class=next(iter(step.condition.required_entities)),
        )
    )
    processed_docs, failed_docs = step([doc])
    assert check_step_has_found_entities(processed_docs[0], step.entity_class) == 8
