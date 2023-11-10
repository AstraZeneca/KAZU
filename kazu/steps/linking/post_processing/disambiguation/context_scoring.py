import logging
from collections.abc import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from kazu.utils.caching import kazu_disk_cache
from kazu.database.in_memory_db import SynonymDatabase
from kazu.utils.utils import create_char_ngrams, create_word_ngrams, Singleton

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
) -> list[str]:
    """Function to create char and word ngrams.

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
    """This class manages a set of TFIDF models (via
    :class:`sklearn.feature_extraction.text.TfidfVectorizer`\\ ).

    It's a singleton, so that the models can be accessed in multiple locations without
    the need to load them into memory multiple times.
    """

    def __init__(self):
        self.synonym_db = SynonymDatabase()
        self.parser_to_vectorizer: dict[str, TfidfVectorizer] = self.build_vectorizers()

    @kazu_disk_cache.memoize(ignore={0})
    def build_vectorizers(self) -> dict[str, TfidfVectorizer]:
        result: dict[str, TfidfVectorizer] = {}
        for parser_name in self.synonym_db.loaded_parsers:
            synonyms = self.synonym_db.get_all(parser_name).keys()
            vectoriser = TfidfVectorizer(lowercase=False, analyzer=create_word_and_char_ngrams)
            vectoriser.fit(synonyms)
            result[parser_name] = vectoriser
        return result

    def __call__(
        self, strings: list[str], matrix: np.ndarray, parser: str
    ) -> Iterable[tuple[str, float]]:
        """Transform a list of strings with a parser-specific vectorizer and score
        against a matrix.

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
