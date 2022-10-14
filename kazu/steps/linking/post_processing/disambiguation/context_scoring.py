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
    """
    function to create char and word ngrams

    :param s: string to process
    :param words: create n words
    :param chars: create n chars
    :return: list of strings comprised of words and chars
    """
    strings = []
    for ngram_word in words:
        strings.extend(create_word_ngrams(s, ngram_word))
    for ngram_char in chars:
        strings.extend(create_char_ngrams(s, ngram_char))
    return strings


class TfIdfScorer(metaclass=Singleton):
    """
    This class manages a set of TFIDF models (via
    :class:`sklearn.feature_extraction.text.TfidfVectorizer`\\ ).

    It's a singleton, so that the models can be accessed in multiple locations without the need to
    load them into memory multiple times.
    """

    def __init__(self, path: Path):
        """

        :param path: to a directory of files containing serialised
            :class:`sklearn.feature_extraction.text.TfidfVectorizer`\\ . The individual filenames
            are used to map the models to the relevant parser
        """
        self.synonym_db = SynonymDatabase()
        self.parser_to_vectorizer: Dict[str, TfidfVectorizer] = {}
        self.build_or_load_vectorizers(path)

    def build_or_load_vectorizers(self, path: Path):
        if path.exists():
            self.load_vectorizers(path)
        else:
            self.build_vectorizers(path)

    def build_vectorizers(self, path: Path):
        path.mkdir(parents=True)
        for parser_name in self.synonym_db.loaded_parsers:
            synonyms = self.synonym_db.get_all(parser_name).keys()
            vectoriser = TfidfVectorizer(lowercase=False, analyzer=create_word_and_char_ngrams)
            vectoriser.fit(synonyms)
            with path.joinpath(parser_name).open(mode="wb") as vectorizer_f:
                pickle.dump(vectoriser, vectorizer_f)
            self.parser_to_vectorizer[parser_name] = vectoriser

    def load_vectorizers(self, path: Path):
        self.parser_to_vectorizer.update(
            {parser_path.name: self.load_vectorizer(parser_path) for parser_path in path.iterdir()}
        )

    @staticmethod
    def load_vectorizer(path: Path) -> TfidfVectorizer:
        with open(path, "rb") as f:
            return pickle.load(f)

    def __call__(
        self, strings: List[str], matrix: np.ndarray, parser: str
    ) -> Iterable[Tuple[str, float]]:
        """
        Transform a list of strings with a parser-specific vectorizer and score against a matrix.

        :param strings:
        :param matrix:
        :param parser:
        :return: matching strings and their score sorted by best score
        """
        if len(strings) == 1:
            yield strings[0], 100.0
        else:
            mat = self.parser_to_vectorizer[parser].transform(strings)
            score_matrix = np.squeeze(-np.asarray(mat.dot(matrix.T).todense()))
            neighbours = score_matrix.argsort()
            for neighbour in neighbours:
                yield strings[neighbour], -score_matrix[neighbour]
