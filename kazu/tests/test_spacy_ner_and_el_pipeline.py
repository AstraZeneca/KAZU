import spacy
import pytest

from hydra.utils import instantiate

from kazu.modelling.ontology_matching.ontology_matcher import OntologyMatcher, SPAN_KEY
from kazu.tests.utils import requires_model_pack


@requires_model_pack
@pytest.fixture(scope="module")
def nlp(kazu_test_config):
    labels = kazu_test_config.RuleBasedNerAndLinkingStep.labels
    syn_table_cache = instantiate(kazu_test_config.RuleBasedNerAndLinkingStep.synonym_table_cache)
    ontologies = syn_table_cache.get_synonym_table_paths()
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    config = {"span_key": SPAN_KEY}
    ontology_matcher = nlp.add_pipe("ontology_matcher", config=config)
    assert isinstance(ontology_matcher, OntologyMatcher)
    ontology_matcher.set_labels(labels)
    ontology_matcher.set_ontologies(ontologies)
    return nlp


# fmt: off
@pytest.mark.parametrize(
    "sentence,entities",
    [
        ("The mean (SD) HAVOC score was 2.6.", []),
        ("Diseases associated with WAS include Wiskott Syndrome and Thrombocytopenia 1.", ["WAS", "Wiskott Syndrome", "Thrombocytopenia 1"]),
        ("MEDI8897 is a recombinant human RSV monoclonal antibody", ["MEDI8897"]),
        ("We aimed to confirm these findings in patients with a BRCA1 or BRCA2 mutation", ["BRCA1", "BRCA2"]),
        ("Gastrointestinal AEs were typically low-grade.", []),
        ("Few clinical trials in asthma have focused on Hispanic populations.", ["asthma"]),
        ("These patients were treated with abemaciclib.", ["abemaciclib"]),
        ("Blood was sampled pre- and post-dose on Day 32.", ["Blood"]),
        ("TIME cells express readily detectable telomerase activity. There is TIME !", ["TIME"]),
        ("Subjects with prevalent kidney disease were randomized to linagliptin or placebo added to usual care.", ["kidney disease", "linagliptin"]),
        ("The increase in lifespan is matched by time free from incident cardiovascular disease.", ["cardiovascular disease", "lifespan"]),
        ("The necuparanib arm had a higher incidence of haematologic toxicity.", ["necuparanib"]),
        ("This was a single-arm trial. My arm hurts.", ["arm"]),
        ("We value life more than anything.", ["life"]),
        ("The main endpoint is quality of life.", []),
        ("The main endpoint is quality-of-life.", []),
        ("IVF, with or without ICSI, was received in all 500 patients.", []),
        ("All three decontamination processes reduced bacteria counts similarly.", []),
        ("The primary endpoint was MFS.", []),
        ("Mean glycated haemoglobin concentration was 66 mmol/mol (8.2%).", []),
        ("Studying pembrolizumab plus neoadjuvant chemotherapy in early-stage breast cancer.", ["pembrolizumab", "breast cancer"]),
        ("Vandetanib plus docetaxel led to a significant improvement in PFS versus placebo plus docetaxel.", ["Vandetanib", "docetaxel", "docetaxel"]),
        ("Anifrolumab dose-dependently suppressed the IFN gene signature.", ["IFN", "Anifrolumab"]),
        ("Antiplatelet effects of citalopram in patients with ischemic stroke", ["citalopram", "ischemic stroke"]),
        ("We reviewed 19 patients with the Dandy-Walker syndrome", ["Dandy-Walker syndrome"]),
        ("We reviewed 19 patients with the Dandy Walker syndrome", ["Dandy Walker syndrome"]),
    ]
)
# fmt: on
def test_ner_results(nlp, sentence, entities):
    doc = nlp(sentence)
    pred_spans = list(doc.spans[SPAN_KEY])
    assert set([s.text for s in pred_spans]) == set(entities)
    assert len(pred_spans) == len(entities)


# fmt: off
@pytest.mark.parametrize(
    "sentence,entities",
    [
        ("We aimed to confirm these findings in patients with a BRCA1 or BRCA2 mutation", ["ENSG00000012048", "ENSG00000139618"]),
        ("These patients were treated with abemaciclib.", ["CHEMBL3301610"]),
        ("Blood was sampled pre- and post-dose on Day 32.", ["http://purl.obolibrary.org/obo/UBERON_0000178"]),
        ("TIME cells express readily detectable telomerase activity", ["CVCL_0047"]),
        ("Studying pembrolizumab plus neoadjuvant chemotherapy in early-stage breast cancer.", ["CHEMBL3137343", "http://purl.obolibrary.org/obo/MONDO_0007254"]),
    ]
)
# fmt: on
def test_nel_results(nlp, sentence, entities):
    doc = nlp(sentence)
    pred_spans = list(doc.spans[SPAN_KEY])
    assert set([s.kb_id_ for s in pred_spans]) == set(entities)
    assert len(pred_spans) == len(entities)
