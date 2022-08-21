from typing import List, Tuple, Optional, Dict, Set, FrozenSet, KeysView

from kazu.data.data import LinkRanks
from kazu.data.data import EquivalentIdSet
from kazu.utils.grouping import sort_then_group
import pandas as pd


class ReactomeDb:

    # reactome:
    # Source database identifier, e.g. UniProt, ENSEMBL, NCBI Gene or ChEBI identifier
    # Reactome Pathway Stable identifier
    # URL
    # Event (Pathway or Reaction) Name
    # Evidence Code
    # Species

    def __init__(self, idx_to_pathways: Dict[str, Set[int]], pathways_to_idx: Dict[int, Set[str]]):
        self.pathways_to_idx = pathways_to_idx
        self.idx_to_pathways = idx_to_pathways

    def get_ids_by_pathway_association(self, query: str) -> List[Set[str]]:
        pathways: Set[int] = self.idx_to_pathways.get(query, set())
        results = []
        for pathway in pathways:
            results.append(self.pathways_to_idx[pathway])
        return results

    def rank_pathways(
        self,
        document_unambiguous_ids: KeysView[Tuple[str, str, LinkRanks]],
        query: FrozenSet[EquivalentIdSet],
    ) -> Optional[str]:
        ranks = []

        for ambiguous_id_set in query:
            for idx in ambiguous_id_set.ids:
                ranks.append(
                    (
                        self.score_pathways(
                            document_unambiguous_ids=document_unambiguous_ids,
                            ambiguous_id=idx,
                        ),
                        idx,
                    ),
                )
        ranks.sort(key=lambda x: x[0], reverse=True)
        top_score, disambiguated_id = ranks[0]
        if top_score > 0.0:
            return disambiguated_id
        else:
            return None

    def score_pathways(
        self, document_unambiguous_ids: KeysView[Tuple[str, str, LinkRanks]], ambiguous_id: str
    ) -> float:
        pathway_sets = sorted(self.get_ids_by_pathway_association(ambiguous_id), reverse=True)
        score = 0.0
        # always prefer confidence values over scores, and return early if a higher confidence result found
        document_unambiguous_ids_by_confidence = sort_then_group(
            document_unambiguous_ids, key_func=lambda x: x[2]
        )

        for link_rank, link_rank_unambiguous in document_unambiguous_ids_by_confidence:
            unambig_ids_this_rank = {x[1] for x in link_rank_unambiguous}
            for pathway in pathway_sets:
                overlap_this_pathway = len(pathway.intersection(unambig_ids_this_rank))
                pathway_score = overlap_this_pathway / len(pathway)
                if pathway_score > score:
                    score = pathway_score
            if score > 0.0:
                break
        return score


def load_kb(path: str) -> ReactomeDb:
    df = pd.read_csv(path, sep="\t")
    df.columns = ["idx", "pathway", "url", "event", "evidence", "species"]
    df = df[["idx", "pathway"]]
    # save some mem by converting to codes
    df["pathway"] = pd.Categorical(df["pathway"]).codes
    df_idx = df.groupby(by="idx").agg(set)
    df_pathway = df.groupby(by="pathway").agg(set)
    return ReactomeDb(df_idx.to_dict()["pathway"], df_pathway.to_dict()["idx"])
