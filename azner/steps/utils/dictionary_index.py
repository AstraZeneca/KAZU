import pandas as pd
from rapidfuzz import process, fuzz


class DictionaryIndex:
    """
    a simple dictionary index for linking
    """

    def __init__(
        self, path: str, name: str = "unnamed_index", fuzzy: bool = False, score_cutoff: int = 99.0
    ):
        """
        :param name: a name for this index
        :param path: a path to a parquet file containing (at least) the columns ['id','default_label','syn']
        :param matching_algorithm: a string describing the string matching algorithm to use. None for direct string
                    match
        """
        self.score_cutoff = score_cutoff
        self.fuzzy = fuzzy
        self.name = name
        self.df = pd.read_parquet(path)

    def search(self, query: str) -> pd.DataFrame:
        """
        search the index
        :param query: a string to query against
        :return: a df of hits
        """
        if not self.fuzzy:
            return self.df[self.df["syn"] == query]
        else:
            hits = process.extract(
                query,
                self.df["syn"].tolist(),
                scorer=fuzz.WRatio,
                limit=2,
                score_cutoff=self.score_cutoff,
            )
            locs = [x[2] for x in hits]
            hit_df = self.df.iloc[locs].copy()
            hit_df["ratio"] = [x[1] for x in hits]
            return hit_df
