import os
from pathlib import Path
from typing import List, Tuple

from data.data import SimpleDocument, Entity, Document

TEST_ASSETS_PATH = Path(__file__).parent.joinpath("test_assets")

TINY_CHEMBL_KB_PATH = TEST_ASSETS_PATH.joinpath("sapbert").joinpath("tiny_chembl.parquet")


def ner_test_cases():
    texts = [
        "EGFR is a gene",
        "CAT1 is a gene",
        "my cat sat on the mat",
        "cat1 is my number plate",
    ]
    return texts


def entity_linking_easy_cases() -> Tuple[List[Document], List[str], List[str]]:
    docs, iris, sources = [], [], []

    doc = SimpleDocument("Baclofen is a muscle relaxant")
    doc.sections[0].entities = [
        Entity(namespace="test", match="Baclofen", entity_class="Drug", start=0, end=8)
    ]
    docs.append(doc)
    iris.append("CHEMBL701")
    sources.append("CHEMBL")

    doc = SimpleDocument("Helium is a gas.")
    doc.sections[0].entities = [
        Entity(namespace="test", match="Helium", entity_class="Drug", start=0, end=6)
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
            match=doc.sections[0].text,
            entity_class="Drug",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("CHEMBL701")
    sources.append("CHEMBL")

    doc = SimpleDocument("Tagrisso")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Drug",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("CHEMBL3353410")
    sources.append("CHEMBL")

    doc = SimpleDocument("Osimertinib")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Drug",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("CHEMBL3353410")
    sources.append("CHEMBL")

    doc = SimpleDocument("Osimertinib Mesylate")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Drug",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("CHEMBL3545063")
    sources.append("CHEMBL")

    doc = SimpleDocument("pain in the head")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Disease",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/MONDO_0021146")
    sources.append("MONDO")

    doc = SimpleDocument("triple-negative breast cancer")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Disease",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/MONDO_0005494")
    sources.append("MONDO")

    doc = SimpleDocument("triple negative cancer of the breast")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Disease",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/MONDO_0005494")
    sources.append("MONDO")

    doc = SimpleDocument("HER2 negative breast cancer")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Disease",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/MONDO_0000618")
    sources.append("MONDO")

    doc = SimpleDocument("HER2 negative cancer")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Disease",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/MONDO_0000618")
    sources.append("MONDO")

    doc = SimpleDocument("bony part of hard palate")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Anatomy",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/UBERON_0012074")
    sources.append("UBERON")

    doc = SimpleDocument("liver")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Anatomy",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/UBERON_0002107")
    sources.append("UBERON")

    doc = SimpleDocument("stomach primordium")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Anatomy",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("http://purl.obolibrary.org/obo/UBERON_0012172")
    sources.append("UBERON")

    doc = SimpleDocument("EGFR")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Gene",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("ENSG00000146648")
    sources.append("ENSEMBL")

    doc = SimpleDocument("epidermal growth factor receptor")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Gene",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("ENSG00000146648")
    sources.append("ENSEMBL")

    doc = SimpleDocument("ERBB1")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Gene",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("ENSG00000146648")
    sources.append("ENSEMBL")

    doc = SimpleDocument("MAPK10")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Gene",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("ENSG00000109339")
    sources.append("ENSEMBL")

    doc = SimpleDocument("mitogen-activated protein kinase 10")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Gene",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("ENSG00000109339")
    sources.append("ENSEMBL")

    doc = SimpleDocument("JNK3")
    doc.sections[0].entities = [
        Entity(
            namespace="test",
            match=doc.sections[0].text,
            entity_class="Gene",
            start=0,
            end=len(doc.sections[0].text),
        )
    ]
    docs.append(doc)
    iris.append("ENSG00000109339")
    sources.append("ENSEMBL")

    return docs, iris, sources


def get_TransformersModelForTokenClassificationNerStep_model_path():
    return os.getenv("TransformersModelForTokenClassificationPath")
