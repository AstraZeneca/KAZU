import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

import numpy as np

from kazu.modelling.database.in_memory_db import SynonymDatabase
from kazu.utils.utils import create_char_ngrams, create_word_ngrams, Singleton
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def create_word_and_char_ngrams(
    s: str,
    words: Iterable[int] = (
        1,
        2,
    ),
    chars: Iterable[int] = (
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
    """
    This class manages a set of TFIDF models (via the :class:`TfIdfDocumentScorer` class) . It's a singleton, so that the
    models can be accessed in multiple locations without the need to load them into memory multiple times
    """

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
        path.mkdir(parents=True)
        for parser_name in self.synonym_db.loaded_parsers:
            synonyms = self.synonym_db.get_all(parser_name).keys()
            scorer = TfIdfDocumentScorer()
            scorer.build_vectoriser(synonyms)
            scorer.save(path.joinpath(parser_name))
            self.parser_to_scorer[parser_name] = scorer

    def load_scorers(self, path: Path):
        self.parser_to_scorer.update({
            parser_path.name: TfIdfDocumentScorer.load(parser_path)
            for parser_path in path.iterdir()
        })


class TfIdfDocumentScorer:
    """
    wrapper class for TfidfVectorizer, to simplify loading/saving/transforming strings
    """

    def __init__(self):
        self.vectoriser: TfidfVectorizer

    def __call__(self, strings: List[str], matrix: np.ndarray) -> Iterable[Tuple[str, float]]:
        """
        transform a list of strings with self.vectoriser and score against a matrix

        :param strings:
        :param matrix:
        :return: matching strings and their score sorted by best score
        """
        if len(strings) == 1:
            yield strings[0], 100.0
        else:
            mat = self.vectoriser.transform(strings)
            score_matrix = np.squeeze(-np.asarray(mat.dot(matrix.T).todense()))
            neighbours = score_matrix.argsort()
            for neighbour in neighbours:
                yield strings[neighbour], -score_matrix[neighbour]

    def transform(self, strings: List[str]) -> np.ndarray:
        return self.vectoriser.transform(strings)

    @staticmethod
    def load(path: Path) -> "TfIdfDocumentScorer":
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def build_vectoriser(self, strings: Iterable[str]):
        self.vectoriser = TfidfVectorizer(lowercase=False, analyzer=create_word_and_char_ngrams)
        self.vectoriser.fit(strings)
