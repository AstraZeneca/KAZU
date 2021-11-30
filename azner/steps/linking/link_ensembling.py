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


class LinkRanks(Enum):
    # labels for ranking
    HIGH_CONFIDENCE = "high_confidence"
    MEDIUM_HIGH_CONFIDENCE = "medium_high_confidence"
    MEDIUM_CONFIDENCE = "medium_confidence"
    LOW_CONFIDENCE = "low_confidence"


def string_overlaps(target: str, query: str):
    """check if target/query overlap in any way"""
    if isinstance(target, str):
        return query in target or target in query
    return False


class MappingPostProcessing:
    """
    ensemble and resolve mapping information for an entity, according to a custom algorithm (see __call__)


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
            data["idx"].append(mapping.idx)
            syn_lower = mapping.metadata.get("syn", None)
            syn_lower = syn_lower.lower() if syn_lower is not None else None
            data["syn"].append(syn_lower)
            label_lower = mapping.metadata.get("default_label", None)
            label_lower = label_lower.lower() if label_lower is not None else None
            data["default_label"].append(label_lower)
            data["mapping"].append(mapping)
            data[LINK_SCORE].append(mapping.metadata[LINK_SCORE])
        self.lookup_df = pd.DataFrame.from_dict(data)

    def exact_hits(self) -> List[Mapping]:
        """
        returns any perfect mappings - i.e. mappings where the match is the same as the default label, synonym, or linker
        score is 100.0 (depends on linkers correctly normalising scores between 0-100). High confidence
        :return:
        """
        hits = (
            self.lookup_df[
                (self.match == self.lookup_df["default_label"])
                | (self.match == self.lookup_df["syn"])
                | (self.lookup_df[LINK_SCORE] == 100.0)
            ]
            .sort_values(by=[LINK_SCORE], ascending=False)["mapping"]
            .tolist()
        )
        self.update_with_confidence(hits, LinkRanks.HIGH_CONFIDENCE.value)
        return hits

    def query_contained_in_hits(self) -> List[Mapping]:
        """
        is the match substring in any of the mappings, or vice versa? Medium confidence
        :return:
        """
        query_length = len(self.match)
        containing_hits = []
        if query_length >= 5:
            containing_hits = (
                self.lookup_df[
                    (
                        self.lookup_df["default_label"].apply(
                            string_overlaps, query=self.match
                        )
                    )
                    | (self.lookup_df["syn"].apply(string_overlaps, query=self.match))
                ]
                .sort_values(by=[LINK_SCORE], ascending=False)["mapping"]
                .tolist()
            )

            self.update_with_confidence(containing_hits, LinkRanks.MEDIUM_CONFIDENCE.value)
        return containing_hits

    def filter_scores(self) -> List[Mapping]:
        """
        do any of the mappings return a score above a threshold for a linker, to suggest a good mapping?
        :return:
        """
        result = []
        namespaces = self.lookup_df[NAMESPACE].unique().tolist()
        for namespace in namespaces:
            filter_score = self.linker_score_thresholds.get(namespace, 100.0)
            hits = (
                self.lookup_df[
                    (self.lookup_df[LINK_SCORE] >= filter_score)
                    & (self.lookup_df[NAMESPACE] == namespace)
                ]
                .sort_values(by=[LINK_SCORE], ascending=False)["mapping"]
                .tolist()
            )
            result.extend(hits)
        self.update_with_confidence(result, LinkRanks.MEDIUM_HIGH_CONFIDENCE.value)
        return result

    def similarly_ranked(self) -> List[Mapping]:
        """
        compare the results of all mapping id's between linkers via set.intersection. Return any that all linkers agree
        on, sorted by score
        :return:
        """
        if len(self.lookup_df[NAMESPACE].unique().tolist()) > 1:
        raw_hits = multiple_hits.groupby(by=["idx"]).filter(lambda x: x[NAMESPACE].nunique() >= 2)
        hits = raw_hits.sort_values(by=[LINK_SCORE], ascending=False)["mapping"].tolist()
        
            self.update_with_confidence(hits, LinkRanks.MEDIUM_HIGH_CONFIDENCE.value)
            return hits
        else:
            return []

    def sort_and_add_confidence(self, df: pd.DataFrame, conf: LinkRanks) -> List[Mapping]:
        mappings = df.sort_values(by=[LINK_SCORE], ascending=False)["mapping"].tolist()
        self.update_with_confidence(mappings, conf)
        return mappings

    @staticmethod
    def update_with_confidence(mappings: List[Mapping], confidence: LinkRanks):
        for mapping in mappings:
            mapping.metadata[LINK_CONFIDENCE] = confidence.value

    def __call__(self):
        """
        preference:
        1) exact hits -> High confidence
        2) scores above threshold for a single linker -> Medium-high confidence
        3) different linkers suggest same idx -> Medium-high confidence
        4) matched text is contained in, or contains mapping default label or synonym -> Medium confidence
        5) best score overall -> Low confidence
        :return:
        """
        hits = self.exact_hits()
        if len(hits) == 0:
            hits = self.filter_scores()
        if len(hits) == 0:
            hits = self.similarly_ranked()
        if len(hits) == 0:
            hits = self.query_contained_in_hits()
        if len(hits) == 0:
            hits = self.sort_scores()
        return hits


class EnsembleEntityLinkingStep(BaseStep):
    """
    ensemble methods to use information from multiple linkers when choosing the 'best' mapping.
    """

    def __init__(
        self,
        depends_on: Optional[List[str]],
        linker_score_thresholds: Dict[str, float],
        keep_top_n: int = 1,
    ):
        super().__init__(depends_on)
        self.linker_score_thresholds = linker_score_thresholds
        self.keep_top_n = keep_top_n

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        for doc in docs:
            try:
                entities = doc.get_entities()
                for entity in entities:
                    if entity.metadata is None or entity.metadata.mappings is None:
                        continue
                    else:
                        processing = MappingPostProcessing(entity, self.linker_score_thresholds)
                        entity.metadata.mappings = processing()[: self.keep_top_n]
            except Exception:
                doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                failed_docs.append(doc)
        return docs, failed_docs
