import logging
import os
import traceback
from typing import Optional, List, Tuple, Callable

from py4j.java_gateway import JavaGateway

from kazu.data.data import Document, PROCESSING_EXCEPTION, Entity
from kazu.steps import BaseStep

logger = logging.getLogger(__name__)


class SethStep(BaseStep):
    """
    A Step that calls SETH (SNP Extraction Tool for Human Variations) over py4j

    @Article{SETH2016,
      Title= {SETH detects and normalizes genetic variants in text.},
      Author= {Thomas, Philippe and Rockt{\"{a}}schel, Tim and Hakenberg, J{\"{o}}rg and Lichtblau, Yvonne and Leser, Ulf},
      Journal= {Bioinformatics},
      Year= {2016},
      Month= {Jun},
      Doi= {10.1093/bioinformatics/btw234},
      Language = {eng},
      Medline-pst = {aheadofprint},
      Pmid = {27256315},
      Url = {http://dx.doi.org/10.1093/bioinformatics/btw234}
    }
    """

    def __init__(
        self,
        depends_on: Optional[List[str]],
        entity_class: str,
        seth_fatjar_path: str,
        condition: Optional[Callable[[Document], bool]] = None,
    ):
        """

        :param depends_on:
        :param entity_class: the entity_class to assign to any Entities that emerge
        :param seth_fatjar_path: path to a py4j fatjar, containing SETH dependencies
        :param condition: Since SETH can be slow, we can optionally specify a callable, so that
            any documents that don't contain pre-existing gene/protein entities are not processed
        """
        super().__init__(depends_on)
        self.condition = condition
        if not os.path.exists(seth_fatjar_path):
            raise RuntimeError(f"required jar: {seth_fatjar_path} not found")
        self.gateway = JavaGateway.launch_gateway(classpath=seth_fatjar_path, die_on_exit=True)
        self.seth = self.gateway.jvm.com.astrazeneca.kazu.SethRunner()
        self.entity_class = entity_class

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        for doc in docs:
            if not self.condition or (self.condition and self.condition(doc)):
                for section in doc.sections:
                    try:
                        mutation_lst = self.seth.findMutations(section.get_text())
                        entities = []
                        for java_mutation_dict in mutation_lst:
                            python_dict = dict(java_mutation_dict)
                            entities.append(
                                Entity.from_spans(
                                    spans=[
                                        (
                                            python_dict.pop("start"),
                                            python_dict.pop("end"),
                                        )
                                    ],
                                    text=section.get_text(),
                                    entity_class=self.entity_class,
                                    namespace=self.namespace(),
                                    metadata=python_dict,
                                ),
                            )
                        section.entities.extend(entities)
                    except Exception:
                        doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                        failed_docs.append(doc)
        return docs, failed_docs
