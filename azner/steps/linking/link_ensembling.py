import traceback
from collections import defaultdict
from enum import Enum
from typing import Optional, List, Tuple, Dict

import pandas as pd

from azner.data.data import (
    Document,
    Mapping,
    LINK_SCORE,
    NAMESPACE,
    LINK_CONFIDENCE,
    Entity,
    PROCESSING_EXCEPTION,
)
from azner.steps import BaseStep
from azner.modelling.ontology_preprocessing.base import DEFAULT_LABEL, IDX, SYN


class LinkRanks(Enum):
    # labels for ranking
    HIGH_CONFIDENCE = "high_confidence"
    MEDIUM_HIGH_CONFIDENCE = "medium_high_confidence"
    MEDIUM_CONFIDENCE = "medium_confidence"
    LOW_CONFIDENCE = "low_confidence"


def string_subsumes(target: str, query: str):
    """check if target/query subsume each other in any way"""
    if isinstance(target, str):
        return query in target or target in query
    return False


class MappingPostProcessing:
    """
    Ensemble and resolve mapping information for an entity, according to a custom algorithm:
    1) exact string match hit in ontology -> High confidence
    2) scores above threshold for a single linker -> Medium-high confidence
    3) different linkers suggest same ontology id -> Medium-high confidence
    4) matched text is contained in, or contains mapping default label or synonym -> Medium confidence
    5) best score overall, across all linkers -> Low confidence
    """

    def __init__(self, entity: Entity, linker_score_thresholds: Dict[str, float]):
        """

        :param entity: the entity to process
        :param linker_score_thresholds:  a dict of Linker Step name, and score threshold. Used by filter_scores
        """
        self.match = entity.match.lower()
        self.linker_score_thresholds = linker_score_thresholds
        self.mappings = entity.metadata.mappings
        data = defaultdict(list)
        for mapping in self.mappings:
            data[NAMESPACE].append(mapping.metadata[NAMESPACE])
            data[IDX].append(mapping.idx)
            syn_lower = mapping.metadata.get(SYN)
            syn_lower = syn_lower.lower() if syn_lower is not None else None
            data[SYN].append(syn_lower)
            label_lower = mapping.metadata.get(DEFAULT_LABEL)
            label_lower = label_lower.lower() if label_lower is not None else None
            data[DEFAULT_LABEL].append(label_lower)
            data["mapping"].append(mapping)
            data[LINK_SCORE].append(mapping.metadata[LINK_SCORE])
        self.lookup_df = pd.DataFrame.from_dict(data)

    def exact_hits(self) -> List[Mapping]:
        """
        returns any perfect mappings - i.e. mappings where the match is the same as the default label, synonym, or linker
        score is 100.0 (depends on linkers correctly normalising scores between 0-100). High confidence
        :return:
        """
        hits = self.lookup_df[
            (self.match == self.lookup_df[DEFAULT_LABEL])
            | (self.match == self.lookup_df[SYN])
            | (self.lookup_df[LINK_SCORE] == 100.0)
        ]
        hits = self.sort_and_add_confidence(hits, LinkRanks.HIGH_CONFIDENCE)
        return hits

    def filter_scores(self) -> List[Mapping]:
        """
        do any of the mappings return a score above a threshold for a linker, to suggest a good mapping?
        :return:
        """
        relevant_linker_thresholds = self.lookup_df[NAMESPACE].map(
            lambda x: self.linker_score_thresholds.get(x, 100.0)
        )
        hits = self.lookup_df[self.lookup_df[LINK_SCORE] >= relevant_linker_thresholds]
        hits = self.sort_and_add_confidence(hits, LinkRanks.MEDIUM_HIGH_CONFIDENCE)
        return hits

    def similarly_ranked(self) -> List[Mapping]:
        """
        compare the results of all mapping id's between linkers via set.intersection. Return any that all linkers agree
        on, sorted by score
        :return:
        """
        if len(self.lookup_df[NAMESPACE].unique().tolist()) > 1:
            raw_hits = self.lookup_df.groupby(by=[IDX]).filter(
                lambda x: x[NAMESPACE].nunique() >= 2
            )
            hits = self.sort_and_add_confidence(raw_hits, LinkRanks.MEDIUM_HIGH_CONFIDENCE)
            return hits
        else:
            return []

    def query_contained_in_hits(self) -> List[Mapping]:
        """
        is the match substring in any of the mappings, or vice versa? Medium confidence. Note, we only check
        matches of 5 chars or more, otherwise will be noisy
        :return:
        """
        query_length = len(self.match)
        hits = []
        if query_length >= 5:
            containing_hits = self.lookup_df[
                (self.lookup_df[DEFAULT_LABEL].apply(string_subsumes, query=self.match))
                | (self.lookup_df[SYN].apply(string_subsumes, query=self.match))
            ]

            hits = self.sort_and_add_confidence(containing_hits, LinkRanks.MEDIUM_CONFIDENCE)
        return hits

    def sort_and_add_confidence(self, df: pd.DataFrame, conf: LinkRanks) -> List[Mapping]:
        mappings = df.sort_values(by=[LINK_SCORE], ascending=False)["mapping"].tolist()
        for mapping in mappings:
            mapping.metadata[LINK_CONFIDENCE] = conf.value
        return mappings

    def __call__(self):
        hits = self.exact_hits()
        if len(hits) == 0:
            hits = self.filter_scores()
        if len(hits) == 0:
            hits = self.similarly_ranked()
        if len(hits) == 0:
            hits = self.query_contained_in_hits()
        if len(hits) == 0:
            hits = self.sort_and_add_confidence(self.lookup_df, LinkRanks.LOW_CONFIDENCE)
        return hits


class EnsembleEntityLinkingStep(BaseStep):
    """
    ensemble methods to use information from multiple linkers when choosing the 'best' mapping. See
    :class:`azner.steps.linking.link_ensembling.MappingPostProcessing`
    """

    def __init__(
        self,
        depends_on: Optional[List[str]],
        linker_score_thresholds: Dict[str, float],
        keep_top_n: int = 1,
    ):
        """

        :param depends_on:
        :param linker_score_thresholds: Dict that maps a linker namespace to it's score threshold
        :param keep_top_n:
        """
        super().__init__(depends_on)
        self.linker_score_thresholds = linker_score_thresholds
        self.keep_top_n = keep_top_n

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        for doc in docs:
            try:
                entities = doc.get_entities()
                for entity in entities:
                    processing = MappingPostProcessing(entity, self.linker_score_thresholds)
                    entity.metadata.mappings = processing()[: self.keep_top_n]
            except Exception:
                doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                failed_docs.append(doc)
        return docs, failed_docs
