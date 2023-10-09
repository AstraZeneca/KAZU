import json
from os import getenv
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest
from kazu.data.data import (
    EquivalentIdSet,
    EquivalentIdAggregationStrategy,
    SynonymTermWithMetrics,
    GlobalParserActions,
    CuratedTerm,
    DocumentJsonUtils,
)
from kazu.language.string_similarity_scorers import StringSimilarityScorer
from kazu.ontology_preprocessing.base import (
    IDX,
    DEFAULT_LABEL,
    SYN,
    MAPPING_TYPE,
    OntologyParser,
)
from kazu.ontology_preprocessing.synonym_generation import CombinatorialSynonymGenerator

TEST_ASSETS_PATH = Path(__file__).parent.joinpath("test_assets")

BERT_TEST_MODEL_PATH = TEST_ASSETS_PATH.joinpath("bert_test_model")

CONFIG_DIR = Path(__file__).parent.parent.joinpath("conf")

SKIP_MESSAGE_NO_MODEL_PACK = """Skipping acceptance test as KAZU_MODEL_PACK is not provided as an \
environment variable.
This should be the path to the kazu model pack root.
"""

requires_model_pack = pytest.mark.skipif(
    getenv("KAZU_MODEL_PACK") is None, reason=SKIP_MESSAGE_NO_MODEL_PACK
)


SKIP_MESSAGE_NO_LABEL_STUDIO = """Skipping acceptance test as either LS_PROJECT_NAME, LS_URL_PORT \
or LS_TOKEN are not provided as an environment variable.
This should indicate the project name and connection/auth information required to retrieve annotations from a \
Label Studio server where the gold standard annotations are stored.
"""

requires_label_studio = pytest.mark.skipif(
    any(getenv(varname) is None for varname in ("LS_PROJECT_NAME", "LS_URL_PORT", "LS_TOKEN")),
    reason=SKIP_MESSAGE_NO_LABEL_STUDIO,
)


def ner_long_document_test_cases() -> list[tuple[str, int, str]]:
    """
    should return list of tuples: 0 = the text, 1 = the number of times an entity class is expected to be found,
    2 = the entity class type
    """
    texts = [
        ("EGFR is a gene, that is also mentioned in this very long document. " * 300, 300, "gene")
    ]
    return texts


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
        data: Optional[dict[str, list[str]]] = None,
        curations_path: Optional[str] = None,
        global_actions: Optional[GlobalParserActions] = None,
    ):
        super().__init__(
            in_path,
            entity_class,
            name,
            string_scorer,
            synonym_merge_threshold,
            data_origin,
            synonym_generator,
            curations_path=curations_path,
            global_actions=global_actions,
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
    ids: list[str],
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
            (
                EquivalentIdSet(
                    ids_and_source=frozenset(
                        (
                            id_,
                            "test",
                        )
                        for id_ in ids
                    )
                ),
            ),
        ),
        aggregated_by=EquivalentIdAggregationStrategy.NO_STRATEGY,
        is_symbolic=True,
        mapping_types=frozenset(),
    )


def write_curations(path: Path, terms: list[CuratedTerm]):
    with open(path, "w") as f:
        for curation in terms:
            f.write(json.dumps(DocumentJsonUtils.obj_to_dict_repr(curation)) + "\n")
