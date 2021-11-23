from typing import Tuple, List

import pandas as pd
from rapidfuzz import process, fuzz

from azner.data.data import LINK_SCORE


class DictionaryIndex:
    """
    a simple dictionary index for linking
    """

    def __init__(
        self, path: str, name: str = "unnamed_index", fuzzy: bool = False, score_cutoff: int = 99.0
    ):
        """

        :param path: path to parquet file of synonyms
        :param name: a name for this index
        :param fuzzy: use fuzzy matching
        :param score_cutoff: minimum score for fuzzy matching
        """
        self.score_cutoff = score_cutoff
        self.fuzzy = fuzzy
        self.name = name
        self.df = pd.read_parquet(path)
        self.df["syn"] = self.df["syn"].str.lower()
        self.lengths = self.df["syn"].apply(len)

    def slice_df_for_potential_syns(self, query: str) -> pd.DataFrame:
        """
        dont attempt string match on any syns that are shorter than the actual query
        returns a slice of self.df for equal or longer syns
        :param query:
        :return:
        """
        query_len = len(query)
        target_indices = self.lengths >= query_len
        df_slice = self.df[target_indices]
        return df_slice

    def post_process_hits(self, query: str, hits: List[Tuple[str, float, int]]):
        """
        prefer exact matching hits, then containing hits
        :param query:
        :param hits:
        :return:
        """

        hit_strings = {x[0]: x for x in hits}
        exact_hits = {x: i for i, x in enumerate(hit_strings) if query == x}
        if len(exact_hits) > 0:
            return [hits[i] for i in exact_hits.values()]
        # ignore any queries shorter than 5 chars as likely to be noisy
        query_length = len(query)
        containing_hits = {
            x: i for i, x in enumerate(hit_strings) if (query_length >= 5 and query in x)
        }
        if len(containing_hits) > 0:
            containing_tups = [hits[i] for i in containing_hits.values()]
            return sorted(containing_tups, key=lambda x: x[1], reverse=True)
        else:
            # fail if no matches found
            return []

    def search(self, query: str) -> pd.DataFrame:
        """
        search the index
        :param query: a string to query against
        :return: a df of hits
        """
        query = query.lower()
        if not self.fuzzy:
            return self.df[self.df["syn"] == query]
        else:
            targets = self.slice_df_for_potential_syns(query)
            hits = process.extract(
                query,
                targets["syn"].tolist(),
                scorer=fuzz.WRatio,
                limit=10,
                score_cutoff=self.score_cutoff,
            )
            hits = self.post_process_hits(query, hits)
            locs = [x[2] for x in hits]
            hit_df = targets.iloc[locs].copy()
            hit_df[LINK_SCORE] = [x[1] for x in hits]
            return hit_df
