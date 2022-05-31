import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytest

from kazu.data.data import Document, Entity, Mapping, CharSpan
from kazu.modelling.ontology_preprocessing.base import (
    IDX,
    DEFAULT_LABEL,
    SYN,
    MAPPING_TYPE,
    OntologyParser,
)

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
        mappings = []
        self.callcount += 1
        return mappings


def full_pipeline_test_cases() -> Tuple[List[Document], List[pd.DataFrame]]:
    docs = []
    dfs = []
    for test_text_path in FULL_PIPELINE_ACCEPTANCE_TESTS_DOCS.glob("*.txt"):
        with test_text_path.open(mode="r") as f:
            text = f.read()
            # .read leaves a final newline if there is one at the end of the file
            # as is standard in a unix file
            assert text[-1] == "\n"
        doc = Document.create_simple_document(text[:-1])
        test_results_path = test_text_path.with_suffix(".csv")
        df = pd.read_csv(test_results_path)
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

    doc = Document.create_simple_document("Baclofen is a muscle relaxant")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match="Baclofen",
            entity_class="drug",
            spans=frozenset([CharSpan(start=0, end=8)]),
        )
    ]
    docs.append(doc)
    iris.append("CHEMBL701")
    sources.append("CHEMBL")

    doc = Document.create_simple_document("Helium is a gas.")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match="Helium",
            entity_class="drug",
            spans=frozenset([CharSpan(start=0, end=6)]),
        )
    ]
    iris.append("CHEMBL1796997")
    sources.append("CHEMBL")
    docs.append(doc)
    return docs, iris, sources


def entity_linking_hard_cases() -> Tuple[List[Document], List[str], List[str]]:
    docs, iris, sources = [], [], []

    doc = Document.create_simple_document("Lioresal")
    add_whole_document_entity(doc, "drug")
    docs.append(doc)
    iris.append("CHEMBL701")
    sources.append("CHEMBL")

    doc = Document.create_simple_document("Tagrisso")
    add_whole_document_entity(doc, "drug")
    docs.append(doc)
    iris.append("CHEMBL3353410")
    sources.append("CHEMBL")

    doc = Document.create_simple_document("Osimertinib")
    add_whole_document_entity(doc, "drug")
    docs.append(doc)
    iris.append("CHEMBL3353410")
    sources.append("CHEMBL")

    doc = Document.create_simple_document("Osimertinib Mesylate")
    add_whole_document_entity(doc, "drug")
    docs.append(doc)
    iris.append("CHEMBL3545063")
    sources.append("CHEMBL")

    doc = Document.create_simple_document("pain in the head")
    add_whole_document_entity(doc, "disease")
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/MONDO_0021146")
    sources.append("MONDO")

    doc = Document.create_simple_document("triple-negative breast cancer")
    add_whole_document_entity(doc, "disease")
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/MONDO_0005494")
    sources.append("MONDO")

    doc = Document.create_simple_document("triple negative cancer of the breast")
    add_whole_document_entity(doc, "disease")
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/MONDO_0005494")
    sources.append("MONDO")

    doc = Document.create_simple_document("HER2 negative breast cancer")
    add_whole_document_entity(doc, "disease")
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/MONDO_0000618")
    sources.append("MONDO")

    doc = Document.create_simple_document("HER2 negative cancer")
    add_whole_document_entity(doc, "disease")
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/MONDO_0000618")
    sources.append("MONDO")

    doc = Document.create_simple_document("bony part of hard palate")
    add_whole_document_entity(doc, "anatomy")
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/UBERON_0012074")
    sources.append("UBERON")

    doc = Document.create_simple_document("liver")
    add_whole_document_entity(doc, "anatomy")
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/UBERON_0002107")
    sources.append("UBERON")

    doc = Document.create_simple_document("stomach primordium")
    add_whole_document_entity(doc, "anatomy")
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/UBERON_0012172")
    sources.append("UBERON")

    doc = Document.create_simple_document("EGFR")
    add_whole_document_entity(doc, "gene")
    docs.append(doc)
    iris.append("ENSG00000146648")
    sources.append("ENSEMBL")

    doc = Document.create_simple_document("epidermal growth factor receptor")
    add_whole_document_entity(doc, "gene")
    docs.append(doc)
    iris.append("ENSG00000146648")
    sources.append("ENSEMBL")

    doc = Document.create_simple_document("ERBB1")
    add_whole_document_entity(doc, "gene")
    docs.append(doc)
    iris.append("ENSG00000146648")
    sources.append("ENSEMBL")

    doc = Document.create_simple_document("MAPK10")
    add_whole_document_entity(doc, "gene")
    docs.append(doc)
    iris.append("ENSG00000109339")
    sources.append("ENSEMBL")

    doc = Document.create_simple_document("mitogen-activated protein kinase 10")
    add_whole_document_entity(doc, "gene")
    docs.append(doc)
    iris.append("ENSG00000109339")
    sources.append("ENSEMBL")

    doc = Document.create_simple_document("JNK3")
    add_whole_document_entity(doc, "gene")
    docs.append(doc)
    iris.append("ENSG00000109339")
    sources.append("ENSEMBL")

    return docs, iris, sources


def add_whole_document_entity(doc: Document, entity_class: str):
    doc.sections[0].entities = [
        Entity.load_contiguous_entity(
            namespace="test",
            match=doc.sections[0].get_text(),
            entity_class=entity_class,
            start=0,
            end=len(doc.sections[0].get_text()),
        )
    ]


def get_TransformersModelForTokenClassificationNerStep_model_path():
    return os.getenv("TransformersModelForTokenClassificationPath")


class DummyParser(OntologyParser):
    DUMMY_SOURCE = "test_parser"
    DUMMY_DATA = {
        IDX: ["first", "first", "second", "second", "third", "alpha"],
        DEFAULT_LABEL: ["1", "1", "2", "2", "3", "4"],
        SYN: ["1", "one", "2", "two", "3", "1"],
        MAPPING_TYPE: ["int", "text", "int", "text", "int", "text"],
    }
    name = DUMMY_SOURCE

    def find_kb(self, string: str) -> str:
        return self.DUMMY_SOURCE

    def parse_to_dataframe(self) -> pd.DataFrame:

        return pd.DataFrame.from_dict(self.DUMMY_DATA)
