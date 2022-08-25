import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

import numpy as np

from kazu.modelling.database.in_memory_db import SynonymDatabase
from kazu.utils.link_index import create_char_ngrams
from kazu.utils.utils import create_word_ngrams, Singleton
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def create_word_and_char_ngrams(
    s: str,
    words: Tuple[int, int] = (
        1,
        2,
    ),
    chars: Tuple[int, int] = (
        2,
        3,
    ),
) -> List[str]:
    strings = []
    for ngram_word in words:
        strings.extend(create_word_ngrams(s, ngram_word))
    for ngram_char in chars:
        strings.extend(create_char_ngrams(s, ngram_char))
    return strings


class TfIdfScorerManager(metaclass=Singleton):
    # singleton so we don't have to use multiple instances of same model
    def __init__(self, path: Path):
        self.synonym_db = SynonymDatabase()
        self.parser_to_scorer: Dict[str, TfIdfDocumentScorer] = {}
        self.build_or_load_scorers(path)

    def build_or_load_scorers(self, path: Path):
        if path.exists():
            self.load_scorers(path)
        else:
            self.build_scorers(path)

    def build_scorers(self, path: Path):
        path.mkdir()
        for parser_name in self.synonym_db.loaded_parsers:
            synonyms = list(self.synonym_db.get_all(parser_name).keys())
            scorer = TfIdfDocumentScorer()
            scorer.build_vectoriser(synonyms)
            scorer.save(path.joinpath(parser_name))
        self.load_scorers(path)

    def load_scorers(self, path: Path):
        for parser_path in path.iterdir():
            self.parser_to_scorer[str(parser_path.name)] = TfIdfDocumentScorer.load(parser_path)


class TfIdfDocumentScorer:
    def __init__(self):
        self.vectoriser: TfidfVectorizer

    def __call__(
        self, synonyms: List[str], document_representation: np.ndarray
    ) -> Iterable[Tuple[str, float]]:
        if len(synonyms) == 1:
            yield synonyms[0], 100.0
        else:
            mat = self.vectoriser.transform(synonyms)
            score_matrix = np.squeeze(-np.asarray(mat.dot(document_representation.T).todense()))
            neighbours = score_matrix.argsort()
            for neighbour in neighbours:
                yield synonyms[neighbour], -score_matrix[neighbour]

    def transform(self, strings: List[str]) -> np.ndarray:
        return self.vectoriser.transform(strings)

    @staticmethod
    def load(path: Path) -> "TfIdfDocumentScorer":
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def build_vectoriser(self, synonyms: List[str]):
        self.vectoriser = TfidfVectorizer(lowercase=False, analyzer=create_word_and_char_ngrams)
        self.vectoriser.fit(synonyms)
