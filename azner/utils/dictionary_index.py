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
            hits = process.extract(
                query,
                self.df["syn"].tolist(),
                scorer=fuzz.WRatio,
                limit=20,
                score_cutoff=self.score_cutoff,
            )
            locs = [x[2] for x in hits]
            hit_df = self.df.iloc[locs].copy()
            hit_df[LINK_SCORE] = [x[1] for x in hits]
            return hit_df
