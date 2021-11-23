from typing import Optional, List, Tuple

from azner.data.data import Document, Mapping, LINK_SCORE
from azner.steps import BaseStep


class EnsembleEntityLinkingStep(BaseStep):
    """
    ensemble methods to use information from multiple linkers when choosing the 'best' mapping
    """

    def __init__(
        self, depends_on: Optional[List[str]], keep_top_n: int = 1, method: str = "highest_conf"
    ):
        super().__init__(depends_on)
        self.method = method
        self.keep_top_n = keep_top_n

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        for doc in docs:
            entities = doc.get_entities()
            for entity in entities:
                if entity.metadata is None and entity.metadata.mappings is None:
                    continue
                else:
                    if self.method == "highest_conf":
                        entity.metadata.mappings = self.highest_coonfidence(
                            entity.metadata.mappings
                        )
        return docs, []

    def highest_coonfidence(self, mappings: List[Mapping]):
        return sorted(mappings, key=lambda x: x[LINK_SCORE], reverse=True)[: self.keep_top_n]
