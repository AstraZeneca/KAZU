import os
import tempfile

import pandas as pd
import pydash
from hydra import initialize_config_module, compose
from hydra.utils import instantiate

from azner.tests.utils import entity_linking_easy_cases
from azner.utils.link_index import IDX, DEFAULT_LABEL, SYN, MAPPING_TYPE


def test_dictionary_entity_linking():
    easy_test_docs, iris, sources = entity_linking_easy_cases()
    with tempfile.TemporaryDirectory() as f:
        test_path = os.path.join(f, "test.parquet")
        texts = []
        for doc in easy_test_docs:
            ents = doc.get_entities()
            texts.extend([x.match for x in ents])
        df = pd.DataFrame.from_dict({IDX: iris, DEFAULT_LABEL: iris, SYN: texts})
        df[MAPPING_TYPE] = [["inferred"] for _ in range(df.shape[0])]
        df.to_parquet(test_path)

        with initialize_config_module(config_module="azner.conf"):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"DictionaryEntityLinkingStep.ontology_dictionary_index.CHEMBL.path={test_path}",
                    "DictionaryEntityLinkingStep.ontology_dictionary_index.CHEMBL.fuzzy=true",
                    "DictionaryEntityLinkingStep.ontology_dictionary_index.CHEMBL.name=CHEMBL",
                    "~DictionaryEntityLinkingStep.ontology_dictionary_index.MONDO",
                    "~DictionaryEntityLinkingStep.ontology_dictionary_index.UBERON",
                    "~DictionaryEntityLinkingStep.ontology_dictionary_index.ENSEMBL",
                ],
            )

            step = instantiate(cfg.DictionaryEntityLinkingStep)
            easy_test_docs, iris, sources = entity_linking_easy_cases()
            successes, failures = step(easy_test_docs)
            entities = pydash.flatten([x.get_entities() for x in successes])
            for entity, iri, source in zip(entities, iris, sources):
                assert entity.metadata.mappings[0].idx == iri
                assert entity.metadata.mappings[0].source == source

            # test cache
            for _ in range(1000):
                easy_test_docs, iris, sources = entity_linking_easy_cases()
                successes, failures = step(easy_test_docs)
                entities = pydash.flatten([x.get_entities() for x in successes])
                for entity, iri, source in zip(entities, iris, sources):
                    assert entity.metadata.mappings[0].idx == iri
                    assert entity.metadata.mappings[0].source == source
