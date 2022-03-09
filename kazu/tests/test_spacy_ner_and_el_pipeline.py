import pytest

from hydra.utils import instantiate

from kazu.modelling.ontology_matching.ontology_matcher import SPAN_KEY
from kazu.modelling.ontology_matching.assemble_pipeline import main as assemble_pipeline
from kazu.tests.utils import requires_model_pack


@requires_model_pack
@pytest.fixture(scope="module")
def nlp(kazu_test_config, tmp_path_factory):
    labels = kazu_test_config.ExplosionNERStep.labels
    syn_table_cache = instantiate(kazu_test_config.ExplosionNERStep.synonym_table_cache)
    parquet_files = syn_table_cache.get_synonym_table_paths()
    return assemble_pipeline(
        parquet_files=parquet_files,
        labels=labels,
        # we don't want this to overwrite the pipeline in the actual model pack
        output_dir=tmp_path_factory.mktemp("pipeline"),
        span_key=SPAN_KEY,
    )


# fmt: off
@pytest.mark.parametrize(
    "sentence,entities",
    [
        ("The mean (SD) HAVOC score was 2.6.", []),
        ("Diseases associated with WAS include Wiskott Syndrome and Thrombocytopenia 1.", ["WAS", "Wiskott Syndrome", "Thrombocytopenia", "Thrombocytopenia 1"]),
        ("MEDI8897 is a recombinant human RSV monoclonal antibody", ["MEDI8897"]),
        ("We aimed to confirm these findings in patients with a BRCA1 or BRCA2 mutation", ["BRCA1", "BRCA2"]),
        ("Gastrointestinal AEs were typically low-grade.", []),
        ("Few clinical trials in asthma have focused on Hispanic populations.", ["asthma"]),
        ("These patients were treated with abemaciclib.", ["abemaciclib"]),
        ("Blood was sampled pre- and post-dose on Day 32.", ["Blood"]),
        ("TIME cells express readily detectable telomerase activity. There is TIME !", ["TIME"]),
        ("Subjects with prevalent kidney disease were randomized to linagliptin or placebo added to usual care.", ["kidney", "kidney disease", "linagliptin"]),
        ("The increase in lifespan is matched by time free from incident cardiovascular disease.", ["cardiovascular disease", "lifespan"]),
        ("The necuparanib arm had a higher incidence of haematologic toxicity.", ["necuparanib"]),
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
        ("Anifrolumab dose-dependently suppressed the IFN gene signature.", ["IFN", "Anifrolumab"]),
        ("Antiplatelet effects of citalopram in patients with ischemic stroke", ["citalopram", "ischemic stroke", "stroke"]),
        ("We reviewed 19 patients with the Dandy-Walker syndrome", ["Dandy-Walker syndrome"]),
        ("We reviewed 19 patients with the Dandy Walker syndrome", ["Dandy Walker syndrome"]),
    ]
)
# fmt: on
def test_ner_results(nlp, sentence, entities):
    doc = nlp(sentence)
    pred_spans = list(doc.spans[SPAN_KEY])
    assert set([s.text for s in pred_spans]) == set(entities)


# fmt: off
@pytest.mark.parametrize(
    "sentence,entities",
    [
        ("We aimed to confirm these findings in patients with a BRCA1 or BRCA2 mutation", ["ENSG00000012048", "ENSG00000139618"]),
        ("These patients were treated with abemaciclib.", ["CHEMBL3301610"]),
        ("Blood was sampled pre- and post-dose on Day 32.", ["http://purl.obolibrary.org/obo/UBERON_0000178"]),
        ("TIME cells express readily detectable telomerase activity", ["CVCL_0047"]),
        (
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
        ),
    ]
)
# fmt: on
def test_nel_results(nlp, sentence, entities):
    doc = nlp(sentence)
    pred_spans = list(doc.spans[SPAN_KEY])
    assert set([s.kb_id_ for s in pred_spans]) == set(entities)
    assert len(pred_spans) == len(entities)
