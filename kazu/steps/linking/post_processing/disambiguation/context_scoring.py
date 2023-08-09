import functools
import json
import logging
from os import getenv
from typing import cast, Optional
from collections.abc import Iterable

import joblib
import numpy as np
from kazu.data.data import EquivalentIdSet
from kazu.database.in_memory_db import SynonymDatabase
from kazu.utils.caching import kazu_disk_cache
from kazu.utils.utils import create_char_ngrams, create_word_ngrams, Singleton
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


class GildaTfIdfScorer(metaclass=Singleton):
    """This class uses a single TFIDF model for 'Gilda-inspired' method of
    disambiguation. It uses a pretrained TF-IDF model, and contexual text
    mapped to knowledgebase identifiers (such as wikipedia descriptions of the
    entity). The sparse matricies of these contexts are then compared cosine
    wise with a target matrix to determine the most likely identifier.

    Context matrices are kept in a disk cache until needed, with only a sample held in memory.
    The size of this in memory cache can be controlled with the
    KAZU_GILDA_TFIDF_DISAMBIGUATION_IN_MEMORY_CACHE_SIZE env variable

    original credit:

    Gyori, Benjamin & Hoyt, Charles & Steppi, Albert. (2021).
    Gilda: biomedical entity text normalization with machine-learned disambiguation as a service.
    10.1101/2021.09.10.459803.

    It's a singleton, so that the model can be accessed in multiple locations without the need to
    load it into memory multiple times.
    """

    def _cache_key(self, parser_name: str, idx: str) -> str:
        return f"{self.__class__.__name__}_{parser_name}_{idx}"

    def __init__(self, contexts_path: str, model_path: str):
        """

        :param contexts_path: path to a json file with structure {<parser_name>:{<identifier:{context_text}>}}
        :param model_path: path to a pretrained :class:`sklearn.feature_extraction.text.TfidfVectorizer` model
        """
        self.model_path = model_path
        self.contexts_path = contexts_path
        self.vectorizer: TfidfVectorizer = joblib.load(model_path)
        self.calculate_id_vectors(self.contexts_path)

    @kazu_disk_cache.memoize(ignore={0})
    def calculate_id_vectors(self, contexts_path: str) -> None:
        """Store the context matrices in the cache.

        Since we don't want every single one
        in memory at runtime, we keep them on disk until needed.
        :param contexts_path:
        :return:
        """
        with open(contexts_path, mode="r") as f:
            data: dict[str, dict[str, str]] = json.load(f)
        for parser_name, ids_and_contexts_dict in data.items():
            mat = self.vectorizer.transform(list(ids_and_contexts_dict.values()))
            with kazu_disk_cache as cache:
                for i, idx in enumerate(ids_and_contexts_dict):
                    cache.set(self._cache_key(parser_name, idx), mat[i])

    @functools.lru_cache(
        maxsize=int(getenv("KAZU_GILDA_TFIDF_DISAMBIGUATION_IN_MEMORY_CACHE_SIZE", 200))
    )
    def _in_memory_disk_cache(self, parser_name: str, idx: str) -> Optional[np.ndarray]:
        return cast(
            Optional[np.ndarray],
            kazu_disk_cache.get(self._cache_key(parser_name=parser_name, idx=idx)),
        )

    def _id_context_score(self, parser_name: str, context_vec: np.ndarray, idx: str) -> float:
        maybe_idx_vec = self._in_memory_disk_cache(parser_name=parser_name, idx=idx)
        # if no context is available, the ID automatically scores 0.0
        # the downside of this is that any id's without a context automatically
        # appear at the bottom of any rankings

        if maybe_idx_vec is None:
            return 0.0
        else:
            return cast(
                float,
                cosine_similarity(context_vec, maybe_idx_vec).item(),
            )

    def __call__(
        self, context_vec: np.ndarray, id_sets: set[EquivalentIdSet], parser_name: str
    ) -> Iterable[tuple[str, float]]:
        """Given a context vector, yield the most likely identifiers and their
        score from the given identifiers.

        :param context_vec:
        :param id_sets:
        :param parser_name:
        :return:
        """
        scores = []
        for equiv_id_set in id_sets:
            for idx in equiv_id_set.ids:
                similarity = self._id_context_score(
                    parser_name=parser_name, context_vec=context_vec, idx=idx
                )
                scores.append(
                    (
                        idx,
                        similarity,
                    )
                )

        for score in sorted(scores, key=lambda x: x[1], reverse=True):
            yield score
