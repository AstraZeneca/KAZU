import os
from os.path import basename
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytest

from kazu.data.data import SimpleDocument, Entity, Document, Mapping

TEST_ASSETS_PATH = Path(__file__).parent.joinpath("test_assets")

TINY_CHEMBL_KB_PATH = TEST_ASSETS_PATH.joinpath("sapbert").joinpath("tiny_chembl.parquet")

FULL_PIPELINE_ACCEPTANCE_TESTS_DOCS = TEST_ASSETS_PATH.joinpath("full_pipeline")

BERT_TEST_MODEL_PATH = TEST_ASSETS_PATH.joinpath("bert_test_model")

CONFIG_DIR = Path(__file__).parent.parent.joinpath("conf")

SKIP_MESSAGE = """
skipping acceptance test as KAZU_MODEL_PACK is not provided as an environment variable. This should be the path 
to the kazu model pack root
"""  # noqa

requires_model_pack = pytest.mark.skipif(
    os.environ.get("KAZU_MODEL_PACK") is None, reason=SKIP_MESSAGE
)


class MockedCachedIndexGroup:
    """
    class for mocking a call to CachedIndexGroup.search
    """

    def __init__(self, iris: List[str], sources: List[str]):
        self.iris = iris
        self.sources = sources
        self.callcount = 0

    def mock_search(self, *args, **kwargs):
        mappings = [
            Mapping(
                source=self.sources[self.callcount],
                idx=self.iris[self.callcount],
                mapping_type=["test"],
            )
        ]
        self.callcount += 1
        return mappings


def full_pipeline_test_cases() -> Tuple[List[SimpleDocument], List[pd.DataFrame]]:
    docs = []
    dfs = []
    test_ids = set(
        [basename(x).split(".")[0] for x in os.listdir(FULL_PIPELINE_ACCEPTANCE_TESTS_DOCS)]
    )
    for id in test_ids:
        with open(FULL_PIPELINE_ACCEPTANCE_TESTS_DOCS.joinpath(f"{id}.txt"), "r") as f:
            text = f.read()
            doc = SimpleDocument(text)
            df = pd.read_csv(FULL_PIPELINE_ACCEPTANCE_TESTS_DOCS.joinpath(f"{id}.csv"))
            docs.append(doc)
            dfs.append(df)
    return docs, dfs


def ner_simple_test_cases():
    """
    should return list of tuples: 0 = the text, 1 = the entity class
    :return:
    """
    texts = [
        ("EGFR is a gene", "gene"),
        ("CAT1 is a gene", "gene"),
        ("my cat sat on the mat", "species"),
    ]
    return texts


def ner_long_document_test_cases():
    """
    should return list of tuples: 0 = the text, 1 = the number of times an entity class is expected to be found,
    2 = the entity class type
    :return:
    """
    texts = [
        ("EGFR is a gene, that is also mentioned in this very long document. " * 300, 300, "gene")
    ]
    return texts


def entity_linking_easy_cases() -> Tuple[List[Document], List[str], List[str]]:
    docs, iris, sources = [], [], []

    doc = SimpleDocument("Baclofen is a muscle relaxant")
    doc.sections[0].entities = [
        Entity(namespace="test", match="Baclofen", entity_class="drug", start=0, end=8)
    ]
    docs.append(doc)
    iris.append("CHEMBL701")
    sources.append("CHEMBL")

    doc = SimpleDocument("Helium is a gas.")
    doc.sections[0].entities = [
        Entity(namespace="test", match="Helium", entity_class="drug", start=0, end=6)
    ]
    iris.append("CHEMBL1796997")
    sources.append("CHEMBL")
    docs.append(doc)
    return docs, iris, sources


def entity_linking_hard_cases() -> Tuple[List[Document], List[str], List[str]]:
    docs, iris, sources = [], [], []

    doc = SimpleDocument("Lioresal")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="drug",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("CHEMBL701")
    sources.append("CHEMBL")

    doc = SimpleDocument("Tagrisso")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="drug",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("CHEMBL3353410")
    sources.append("CHEMBL")

    doc = SimpleDocument("Osimertinib")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="drug",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("CHEMBL3353410")
    sources.append("CHEMBL")

    doc = SimpleDocument("Osimertinib Mesylate")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="drug",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("CHEMBL3545063")
    sources.append("CHEMBL")

    doc = SimpleDocument("pain in the head")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="disease",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/MONDO_0021146")
    sources.append("MONDO")

    doc = SimpleDocument("triple-negative breast cancer")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="disease",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/MONDO_0005494")
    sources.append("MONDO")

    doc = SimpleDocument("triple negative cancer of the breast")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="disease",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/MONDO_0005494")
    sources.append("MONDO")

    doc = SimpleDocument("HER2 negative breast cancer")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="disease",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/MONDO_0000618")
    sources.append("MONDO")

    doc = SimpleDocument("HER2 negative cancer")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="disease",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/MONDO_0000618")
    sources.append("MONDO")

    doc = SimpleDocument("bony part of hard palate")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="anatomy",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/UBERON_0012074")
    sources.append("UBERON")

    doc = SimpleDocument("liver")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="anatomy",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/UBERON_0002107")
    sources.append("UBERON")

    doc = SimpleDocument("stomach primordium")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="anatomy",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/UBERON_0012172")
    sources.append("UBERON")

    doc = SimpleDocument("EGFR")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="gene",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("ENSG00000146648")
    sources.append("ENSEMBL")

    doc = SimpleDocument("epidermal growth factor receptor")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="gene",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("ENSG00000146648")
    sources.append("ENSEMBL")

    doc = SimpleDocument("ERBB1")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="gene",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("ENSG00000146648")
    sources.append("ENSEMBL")

    doc = SimpleDocument("MAPK10")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="gene",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("ENSG00000109339")
    sources.append("ENSEMBL")

    doc = SimpleDocument("mitogen-activated protein kinase 10")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="gene",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("ENSG00000109339")
    sources.append("ENSEMBL")

    doc = SimpleDocument("JNK3")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class="gene",
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]
    docs.append(doc)
    iris.append("ENSG00000109339")
    sources.append("ENSEMBL")

    return docs, iris, sources


def get_TransformersModelForTokenClassificationNerStep_model_path():
    return os.getenv("TransformersModelForTokenClassificationPath")
