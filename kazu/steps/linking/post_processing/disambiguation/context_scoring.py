import functools
import json
import logging
import pickle
from collections.abc import Iterable
from os import getenv
from pathlib import Path
from typing import cast, Optional

import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import safe_sparse_dot

from kazu.data import EquivalentIdSet
from kazu.database.in_memory_db import SynonymDatabase
from kazu.utils.caching import kazu_disk_cache
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


class GildaTfIdfScorer(metaclass=Singleton):
    """This class uses a single TFIDF model for 'Gilda-inspired' method of
    disambiguation. It uses a pretrained TF-IDF model, and contexual text mapped to
    knowledgebase identifiers (such as wikipedia descriptions of the entity). The sparse
    matrices of these contexts are then compared cosine wise with a target matrix to
    determine the most likely identifier.

    Context matrices are kept in a disk cache until needed, with only a sample held in memory.
    The size of this in memory cache can be controlled with the
    ``KAZU_GILDA_TFIDF_DISAMBIGUATION_IN_MEMORY_CACHE_SIZE`` env variable.

    .. caution::
       If no context is available, the ID automatically scores 0.0.
       The downside of this is that any ids without a context automatically
       appear at the bottom of any rankings.

    Original Credit:

    https://github.com/indralab/gilda

    Paper:

    | Benjamin M Gyori, Charles Tapley Hoyt, and Albert Steppi. 2022.
    | `Gilda: biomedical entity text normalization with machine-learned disambiguation as a service. <https://doi.org/10.1093/bioadv/vbac034>`_
    | Bioinformatics Advances. Vbac034.

    .. raw:: html

        <details>
        <summary>Bibtex Citation Details</summary>

    .. code:: bibtex

        @article{gyori2022gilda,
            author = {Gyori, Benjamin M and Hoyt, Charles Tapley and Steppi, Albert},
            title = "{{Gilda: biomedical entity text normalization with machine-learned disambiguation as a service}}",
            journal = {Bioinformatics Advances},
            year = {2022},
            month = {05},
            issn = {2635-0041},
            doi = {10.1093/bioadv/vbac034},
            url = {https://doi.org/10.1093/bioadv/vbac034},
            note = {vbac034}
        }

    .. raw:: html

        </details>

    It's a singleton, so that the model can be accessed in multiple locations without the need to
    load it into memory multiple times.
    """

    def _cache_key(self, parser_name: str, idx: str) -> str:
        return f"{self.__class__.__name__}_{parser_name}_{idx}"

    def __init__(self, contexts_path: str, model_path: str):
        """

        :param contexts_path: json file in the format:

            .. code-block:: python

                {"<parser name>": {"<idx>": "<context string>"}}

        :param model_path: path to a pretrained :class:`sklearn.feature_extraction.text.TfidfVectorizer` model
        """
        self.model_path = model_path
        self.contexts_path = contexts_path
        with open(model_path, mode="rb") as f:
            self.vectorizer: TfidfVectorizer = pickle.load(f)
        contexts_path_as_path = Path(contexts_path)
        self._calculate_id_vectors(
            directory=contexts_path_as_path.parent, filename=contexts_path_as_path.name
        )
        self.null_vector = self.vectorizer.transform([""])

    @kazu_disk_cache.memoize(ignore={0, 1})
    def _calculate_id_vectors(self, directory: Path, filename: str) -> None:
        """Calculate the TF-IDF vectors for the file of contexts.

        This method is disk cached - since we don't want every
        single one in memory at runtime, we keep them on disk
        until needed.

        Note that the directory and filename are specified seperately,
        so that disk caching works correctly on different machines.

        :param directory: Directory containing contexts file
        :param filename: contexts json file (see __init__ for format)
        :return:
        """
        with directory.joinpath(filename).open(mode="r") as f:
            data: dict[str, dict[str, str]] = json.load(f)
        for parser_name, ids_and_contexts_dict in data.items():
            mat = self.vectorizer.transform(list(ids_and_contexts_dict.values()))
            with kazu_disk_cache as cache:
                for i, idx in enumerate(ids_and_contexts_dict):
                    cache.set(self._cache_key(parser_name, idx), mat[i])

    @functools.lru_cache(
        maxsize=int(getenv("KAZU_GILDA_TFIDF_DISAMBIGUATION_IN_MEMORY_CACHE_SIZE", 200))
    )
    def _in_memory_disk_cache(self, parser_name: str, idx: str) -> Optional[csr_matrix]:
        return cast(
            Optional[csr_matrix],
            kazu_disk_cache.get(self._cache_key(parser_name=parser_name, idx=idx)),
        )

    def __call__(
        self, context_vec: np.ndarray, id_sets: set[EquivalentIdSet], parser_name: str
    ) -> Iterable[tuple[str, float]]:
        """Given a context vector, yield the most likely identifiers and their score
        from the given set of identifiers.

        :param context_vec:
        :param id_sets:
        :param parser_name:
        :return: identifier strings and scores, starting with the string with the best
            score
        """
        idx_to_vec = {}
        for equiv_id_set in id_sets:
            for idx in equiv_id_set.ids:
                maybe_id_vec = self._in_memory_disk_cache(parser_name=parser_name, idx=idx)
                if maybe_id_vec is not None:
                    idx_to_vec[idx] = maybe_id_vec
                else:
                    idx_to_vec[idx] = self.null_vector
        if idx_to_vec:
            idx_lst = list(idx_to_vec.keys())
            scores = -(
                safe_sparse_dot(
                    context_vec, vstack(idx_to_vec.values()).T, dense_output=True
                ).squeeze()
            )
            neighbours = scores.argsort()
            distances = scores[neighbours]
            for neighbour, score in zip(neighbours, distances):
                idx = idx_lst[neighbour]
                yield idx, -score
