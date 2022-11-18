from pathlib import Path
from os import getenv
from typing import List, Tuple, Dict, Optional

import pandas as pd
import pytest
from kazu.data.data import (
    Document,
    Entity,
    CharSpan,
    EquivalentIdSet,
    EquivalentIdAggregationStrategy,
    SynonymTermWithMetrics,
)
from kazu.modelling.ontology_preprocessing.base import (
    IDX,
    DEFAULT_LABEL,
    SYN,
    MAPPING_TYPE,
    OntologyParser,
)
from kazu.modelling.language.string_similarity_scorers import StringSimilarityScorer
from kazu.modelling.ontology_preprocessing.synonym_generation import CombinatorialSynonymGenerator

TEST_ASSETS_PATH = Path(__file__).parent.joinpath("test_assets")

TINY_CHEMBL_KB_PATH = TEST_ASSETS_PATH.joinpath("sapbert").joinpath("tiny_chembl.parquet")

FULL_PIPELINE_ACCEPTANCE_TESTS_DOCS = TEST_ASSETS_PATH.joinpath("full_pipeline")

BERT_TEST_MODEL_PATH = TEST_ASSETS_PATH.joinpath("bert_test_model")

CONFIG_DIR = Path(__file__).parent.parent.joinpath("conf")

SKIP_MESSAGE_NO_MODEL_PACK = """
skipping acceptance test as KAZU_MODEL_PACK is not provided as an environment variable. This should be the path 
to the kazu model pack root
"""  # noqa

requires_model_pack = pytest.mark.skipif(
    getenv("KAZU_MODEL_PACK") is None, reason=SKIP_MESSAGE_NO_MODEL_PACK
)


SKIP_MESSAGE_NO_LABEL_STUDIO = """
skipping acceptance test as either LS_PROJECT_NAME, LS_URL_PORT or LS_TOKEN are not provided as an environment variable
This should indicate the project name and connection/auth information required to retrieve annotations from a 
Label Studio server where the gold standard annotations are stored.
"""  # noqa

requires_label_studio = pytest.mark.skipif(
    any(getenv(varname) is None for varname in ("LS_PROJECT_NAME", "LS_URL_PORT", "LS_TOKEN")),
    reason=SKIP_MESSAGE_NO_LABEL_STUDIO,
)


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
    """
    texts = [
        ("EGFR is a gene", "gene"),
        ("CAT1 is a gene", "gene"),
        ("my cat sat on the mat", "species"),
        ("For the treatment of anorexia nervosa.", "disease"),
    ]
    return texts


def ner_long_document_test_cases():
    """
    should return list of tuples: 0 = the text, 1 = the number of times an entity class is expected to be found,
    2 = the entity class type
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
    return getenv("TransformersModelForTokenClassificationPath")


class DummyParser(OntologyParser):
    DEFAULT_DUMMY_DATA = {
        IDX: ["first", "first", "second", "second", "third", "alpha"],
        DEFAULT_LABEL: ["1", "1", "2", "2", "3", "4"],
        SYN: ["1", "one", "2", "two", "3", "4"],
        MAPPING_TYPE: ["int", "text", "int", "text", "int", "int"],
    }

    def __init__(
        self,
        in_path: str = "",
        entity_class: str = "test",
        name: str = "test_parser",
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.7,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        source: str = "test_parser",
        data: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__(
            in_path,
            entity_class,
            name,
            string_scorer,
            synonym_merge_threshold,
            data_origin,
            synonym_generator,
        )
        self.source = source
        if data is not None:
            self.data = data
        else:
            self.data = self.DEFAULT_DUMMY_DATA

    def find_kb(self, string: str) -> str:
        return self.source

    def parse_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.data)


def make_dummy_synonym_term(
    ids: List[str],
    parser_name: str,
    search_score: Optional[float] = None,
    embed_score: Optional[float] = None,
) -> SynonymTermWithMetrics:
    return SynonymTermWithMetrics(
        terms=frozenset(
            (
                "1",
                "one",
            )
        ),
        term_norm="1",
        parser_name=parser_name,
        search_score=search_score,
        embed_score=embed_score,
        associated_id_sets=frozenset(
            (EquivalentIdSet(ids_to_source={id_: "test" for id_ in ids}, ids=frozenset(ids)),)
        ),
        aggregated_by=EquivalentIdAggregationStrategy.NO_STRATEGY,
        is_symbolic=True,
        mapping_types=frozenset(),
    )
