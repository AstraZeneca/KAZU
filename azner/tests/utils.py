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


def entity_linking_easy_cases() -> Tuple[List[Document], List[str]]:
    docs, iris = [], []

    doc = SimpleDocument("Baclofen is a muscle relaxant")
    doc.sections[0].entities = [
        Entity(namespace="test", match="Baclofen", entity_class="Drug", start=0, end=8)
    ]
    docs.append(doc)
    iris.append("http://rdf.ebi.ac.uk/resource/chembl/molecule/CHEMBL701")

    doc = SimpleDocument("Lioresal is another name for baclofen")
    doc.sections[0].entities = [
        Entity(namespace="test", match="Lioresal", entity_class="Drug", start=0, end=8)
    ]
    docs.append(doc)
    iris.append("http://rdf.ebi.ac.uk/resource/chembl/molecule/CHEMBL701")

    doc = SimpleDocument("Helium is a gas.")
    doc.sections[0].entities = [
        Entity(namespace="test", match="Helium", entity_class="Drug", start=0, end=6)
    ]
    iris.append("http://rdf.ebi.ac.uk/resource/chembl/molecule/CHEMBL1796997")
    docs.append(doc)
    return docs, iris


def get_TransformersModelForTokenClassificationNerStep_model_path():
    return os.getenv("TransformersModelForTokenClassificationPath")
