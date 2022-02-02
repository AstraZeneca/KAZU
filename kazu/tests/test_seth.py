from hydra.utils import instantiate

from kazu.data.data import Document
from kazu.tests.utils import (
    requires_model_pack,
)


@requires_model_pack
def test_seth_step(kazu_test_config):
    step = instantiate(kazu_test_config.SethStep)
    doc = Document.create_simple_document(
        "Causative GJB2 mutations were identified in 31 (15.2%) patients, and two common mutations, c.35delG and L90P "
        "(c.269T>C), accounted for 72.1% and 9.8% of GJB2 disease alleles."
    )
    success_docs, failed_docs = step([doc]*10)
    assert len(failed_docs) == 0
    expected_hgvs_strings = ["p.Leu90Pro", "c.269T>C", "c.35delG"]
    found_hgvs_strings = [ent.metadata["hgvs"] for ent in success_docs[0].get_entities()]
    assert all([expected_str in found_hgvs_strings for expected_str in expected_hgvs_strings])
