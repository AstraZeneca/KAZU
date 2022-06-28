from typing import List, Tuple

import pandas as pd
import pydash
import pytest
from hydra.utils import instantiate

from kazu.data.data import Entity, Document
from kazu.pipeline import Pipeline, load_steps
from kazu.steps.ner.explosion import ExplosionNERStep
from kazu.tests.utils import (
    entity_linking_hard_cases,
    requires_model_pack,
    full_pipeline_test_cases,
    ner_simple_test_cases,
    ner_long_document_test_cases,
)

# TODO: we need a much better/consistent evaluation dataset for acceptance tests. Currently these all fail until we have
# one


# this applies the require_model_pack mark to all tests in this module
pytestmark = requires_model_pack


# fails because of switch to using Hits rather than Mappings
@pytest.mark.xfail
def test_sapbert_acceptance(kazu_test_config):
    minimum_pass_score = 0.80
    hits = []
    misses = []
    step = instantiate(kazu_test_config.SapBertForEntityLinkingStep)
    easy_test_docs, iris, sources = entity_linking_hard_cases()
    successes, failures = step(easy_test_docs)
    entities = pydash.flatten([x.get_entities() for x in successes])
    for entity, iri, source in zip(entities, iris, sources):
        if len(entity.mappings) > 0 and entity.mappings[0].idx == iri:
            hits.append(entity)
        else:
            misses.append(
                (
                    entity,
                    iri,
                )
            )

    for entity, iri in misses:
        print(f"missed {entity.match}: got {entity.mappings}, wanted {iri} ")
    total = len(hits) + len(misses)
    score = len(hits) / total
    if score < minimum_pass_score:
        pytest.fail(
            f"sapbert scored {score}, which did not reach minimum pass score of {minimum_pass_score}"
        )


# fails as ExplosionNERStep is now picking up many more entities
@pytest.mark.xfail
def test_full_pipeline_acceptance_test(kazu_test_config):
    pipeline = Pipeline(steps=load_steps(kazu_test_config))
    docs, annotation_dfs = full_pipeline_test_cases()
    successes = pipeline(docs)
    for doc, annotations in zip(successes, annotation_dfs):
        section = doc.sections[0]
        for entity in section.entities:
            matches = query_annotations_df(annotations, entity)
            if matches.shape[0] != 1:
                pytest.fail(f"failed to match {entity} in section: {section.text}")


def query_annotations_df(annotations: pd.DataFrame, entity: Entity):
    if len(entity.mappings) > 0:
        mapping_id = entity.mappings[0].idx
    else:
        mapping_id = None

    matches = annotations[
        (annotations["start"] == entity.start)
        & (annotations["end"] == entity.end)
        & (annotations["match"] == entity.match)
        & (annotations["entity_class"] == entity.entity_class)
        & (
            (annotations["mapping_id"] == mapping_id)
            if mapping_id is not None
            else (pd.isna(annotations["mapping_id"]))
        )
    ]
    return matches


# fails because of move to Hits from Mappings
@pytest.mark.xfail
def test_dictionary_entity_linking(override_kazu_test_config):
    minimum_pass_score = 0.80
    cfg = override_kazu_test_config(overrides=["pipeline=entity_linking_only"])
    hits = []
    misses = []
    pipeline = Pipeline(steps=load_steps(cfg))
    easy_test_docs, iris, sources = entity_linking_hard_cases()
    successes = pipeline(easy_test_docs)
    entities = pydash.flatten([x.get_entities() for x in successes])
    for entity, iri, source in zip(entities, iris, sources):
        if iri in [x.idx for x in entity.mappings]:
            hits.append(entity)
        else:
            misses.append(
                (
                    entity,
                    iri,
                )
            )

    for entity, iri in misses:
        if len(entity.mappings) > 0:
            print(f"missed {entity.match}: got {entity.mappings[0].idx}, wanted {iri} ")
        else:
            print(f"missed {entity.match}: got Nothing, wanted {iri} ")
    total = len(hits) + len(misses)
    score = len(hits) / total
    if score < minimum_pass_score:
        pytest.fail(
            f"sapbert scored {score}, which did not reach minimum pass score of {minimum_pass_score}"
        )


@pytest.mark.xfail
def test_TransformersModelForTokenClassificationNerStep(kazu_test_config):
    step = instantiate(kazu_test_config.TransformersModelForTokenClassificationNerStep)
    simple_test_cases = ner_simple_test_cases()
    docs, classes = [], []
    for text, expected_class in simple_test_cases:
        docs.append(Document.create_simple_document(text))
        classes.append(expected_class)

    successes, failures = step(docs)
    assert len(successes) == len(docs)
    for doc, target_class in zip(successes, classes):
        assert doc.sections[0].entities[0].entity_class == target_class
        assert len(doc.sections[0].entities) == 1, (
            f"there should be a single entity of class {target_class} "
            f"recognised for the text {doc.sections[0]}"
        )
    long_doc_test_cases = ner_long_document_test_cases()
    docs, expected_ent_count, classes = [], [], []
    for text, target_entity_count, expected_class in long_doc_test_cases:
        docs.append(Document.create_simple_document(text))
        classes.append(expected_class)

    successes, failures = step(docs)
    assert len(successes) == len(docs)
    for doc, target_entity_count, target_class in zip(successes, expected_ent_count, classes):
        assert len(doc.sections[0].entities) == target_entity_count, (
            f"there should be n:{target_entity_count} "
            f"entities recognised for the text "
            f"{doc.sections[0]}"
        )
        for ent in doc.sections[0].entities:
            assert ent.entity_class == target_class


@pytest.fixture(scope="module")
def explosion_ner_step(kazu_test_config) -> ExplosionNERStep:
    return instantiate(kazu_test_config.ExplosionNERStep)


# fmt: off
test_ner_sentences_and_entities: List[Tuple[str, List[str]]] = [
    ("The mean (SD) HAVOC score was 2.6.", []),
    ("Diseases associated with WAS include Wiskott Syndrome and Thrombocytopenia 1.", ["WAS", "Wiskott Syndrome", "Thrombocytopenia", "Thrombocytopenia 1"]),
    pytest.param("MEDI8897 is a recombinant human RSV monoclonal antibody", ["MEDI8897"], marks=pytest.mark.xfail),
    ("We aimed to confirm these findings in patients with a BRCA1 or BRCA2 mutation", ["BRCA1", "BRCA2"]),
    ("Gastrointestinal AEs were typically low-grade.", []),
    ("Few clinical trials in asthma have focused on Hispanic populations.", ["asthma"]),
    ("These patients were treated with abemaciclib.", ["abemaciclib"]),
    pytest.param("Blood was sampled pre- and post-dose on Day 32.", ["Blood"], marks=pytest.mark.xfail),
    ("TIME cells express readily detectable telomerase activity. There is TIME !", ["TIME"]),
    ("Subjects with prevalent kidney disease were randomized to linagliptin or placebo added to usual care.", ["kidney", "kidney disease", "linagliptin"]),
    pytest.param("The increase in lifespan is matched by time free from incident cardiovascular disease.", ["cardiovascular disease", "lifespan"], marks=pytest.mark.xfail),
    pytest.param("The necuparanib arm had a higher incidence of haematologic toxicity.", ["necuparanib"], marks=pytest.mark.xfail),
    ("This was a single-arm trial. My arm hurts.", ["arm"]),
    ("We value life more than anything.", ["life"]),
    ("The main endpoint is quality of life.", []),
    ("The main endpoint is quality-of-life.", []),
    ("IVF, with or without ICSI, was received in all 500 patients.", []),
    ("All three decontamination processes reduced bacteria counts similarly.", []),
    ("The primary endpoint was MFS.", []),
    ("Vandetanib plus docetaxel led to a significant improvement in PFS versus placebo plus docetaxel.", ["Vandetanib", "docetaxel", "docetaxel"]),
    ("Mean glycated haemoglobin concentration was 66 mmol/mol (8.2%).", ["haemoglobin"]),
    ("Studying pembrolizumab plus neoadjuvant chemotherapy in early-stage breast cancer.", ["pembrolizumab", "breast cancer", "breast", "cancer", "chemotherapy"]),
    pytest.param("Anifrolumab dose-dependently suppressed the IFN gene signature.", ["IFN", "Anifrolumab"], marks=pytest.mark.xfail),
    ("Antiplatelet effects of citalopram in patients with ischemic stroke", ["citalopram", "ischemic stroke", "stroke"]),
    ("We reviewed 19 patients with the Dandy-Walker syndrome", ["Dandy-Walker syndrome"]),
    ("We reviewed 19 patients with the Dandy Walker syndrome", ["Dandy Walker syndrome"]),
]
# fmt: on


@pytest.mark.parametrize(
    argnames=["sentence", "entities"], argvalues=test_ner_sentences_and_entities
)
@requires_model_pack
def test_ExplosionNERStep_single_ner_results(
    explosion_ner_step: ExplosionNERStep, sentence: str, entities: List[str]
):
    doc = Document.create_simple_document(sentence)
    successes, failures = explosion_ner_step(docs=[doc])
    assert len(successes) == 1 and len(failures) == 0

    pred_ents = successes[0].sections[0].entities
    assert set(e.match for e in pred_ents) == set(entities)


# this fails because of fails to the single test, but also it currently
# throws an error because we try to upack the 'params' where there's an xfail, which
# doesn't work.
@pytest.mark.xfail
@requires_model_pack
def test_ExplosionNERStep_batch_ner_results(explosion_ner_step: ExplosionNERStep):
    """tests that we can feed in multiple docs at once and get the correct response.

    Note, this will only pass if all parametrizations of test_ExplosionNERStep_single_ner_results pass,
    but having the parametrization is beneficial since we see all failures at once, not just the first doc that
    fails and then need to keep re-running tests. But we also want to check we can pass a batch of documents to the
    step and it still work."""
    docs = [
        Document.create_simple_document(sentence)
        for sentence, entities in test_ner_sentences_and_entities
    ]
    entities = [entities for sentence, entities in test_ner_sentences_and_entities]
    successes, failures = explosion_ner_step(docs)
    assert len(successes) == len(docs) and len(failures) == 0
    for doc, ents_for_doc in zip(successes, entities):
        pred_ents = doc.sections[0].entities
        assert set(e.match for e in pred_ents) == set(ents_for_doc)


# these all fail because of move to Hits from mappings, but possibly also for other reasons.
# fmt: off
test_nel_sentences_and_ids: List[Tuple[str, List[str]]] = [
    pytest.param("We aimed to confirm these findings in patients with a BRCA1 or BRCA2 mutation", ["ENSG00000012048", "ENSG00000139618"], marks=pytest.mark.xfail),
    pytest.param("These patients were treated with abemaciclib.", ["CHEMBL3301610"], marks=pytest.mark.xfail),
    pytest.param("Blood was sampled pre- and post-dose on Day 32.", ["http://purl.obolibrary.org/obo/UBERON_0000178"], marks=pytest.mark.xfail),
    pytest.param("TIME cells express readily detectable telomerase activity", ["CVCL_0047"], marks=pytest.mark.xfail),
    pytest.param(
        "Studying pembrolizumab plus neoadjuvant chemotherapy in early-stage breast cancer.",
        [
            "CHEMBL3137343",  # pembrolizumab
            "http://purl.obolibrary.org/obo/MONDO_0007254",  # MONDO breast cancer
            "http://purl.obolibrary.org/obo/UBERON_0000310",  # UBERON breaset
            "http://purl.obolibrary.org/obo/MONDO_0004992",  # MONDO cancer
            "http://purl.obolibrary.org/obo/HP_0002664",  # HP neoplasm (~= cancer)
            # this is 'breast carcinoma' - this one is a bit questionable as http://purl.obolibrary.org/obo/MONDO_0007254 'breast cancer' is better -
            # ideally we would use the synonym type to prefer 'breast cancer' since the link for 'breast carcinoma' is 'has broad synonym'.
            # However, we're not using the entity linking for now, so not worth it.
            "http://purl.obolibrary.org/obo/MONDO_0004989",
            "10006187",  # Meddra breast cancer
            "10028997",  # Meddra Neoplasm malignant (~=cancer)
            "10061758",  # Meddra Chemotherapy

        ],
        marks=pytest.mark.xfail,
    ),
]
# fmt: on


@pytest.mark.parametrize(argnames=["sentence", "ids"], argvalues=test_nel_sentences_and_ids)
@requires_model_pack
def test_ExplosionNERStep_single_nel_results(
    explosion_ner_step: ExplosionNERStep, sentence: str, ids: List[str]
):
    doc = Document.create_simple_document(sentence)
    successes, failures = explosion_ner_step(docs=[doc])
    assert len(successes) == 1 and len(failures) == 0

    pred_ents = successes[0].sections[0].entities
    # TODO: is this the full check we want to do?
    assert set(mapping for e in pred_ents for mapping in e.mappings) == set(ids)


# this fails because of fails to the single test, but also it currently
# throws an error because we try to upack the 'params' where there's an xfail, which
# doesn't work.
@pytest.mark.xfail
@requires_model_pack
def test_ExplosionNERStep_batch_nel_results(explosion_ner_step: ExplosionNERStep):
    """tests that we can feed in multiple docs at once and get the correct response.

    Note, this will only pass if all parametrizations of test_ExplosionNERStep_single_ner_results pass,
    but having the parametrization is beneficial since we see all failures at once, not just the first doc that
    fails and then need to keep re-running tests. But we also want to check we can pass a batch of documents to the
    step and it still work."""
    docs = [
        Document.create_simple_document(sentence) for sentence, ids in test_nel_sentences_and_ids
    ]
    ids = [ids for sentence, ids in test_nel_sentences_and_ids]
    successes, failures = explosion_ner_step(docs)
    assert len(successes) == len(docs) and len(failures) == 0
    for doc, ids_for_doc in zip(successes, ids):
        pred_ents = doc.sections[0].entities
        # TODO: as above
        assert set(mapping for e in pred_ents for mapping in e.mappings) == set(ids_for_doc)
