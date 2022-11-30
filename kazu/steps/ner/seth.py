import os
from typing import Optional, Callable

from py4j.java_gateway import JavaGateway

from kazu.data.data import Document, Entity, Mapping, LinkRanks
from kazu.steps import Step, document_iterating_step


SETH_METADATA_KEY = "seth"


class SethStep(Step):
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
        entity_class: str,
        seth_fatjar_path: str,
        java_home: str,
        condition: Optional[Callable[[Document], bool]] = None,
    ):
        """
        :param entity_class: the entity_class to assign to any Entities that emerge
        :param seth_fatjar_path: path to a py4j fatjar, containing SETH dependencies
        :param condition: Since SETH can be slow, we can optionally specify a callable, so that
            any documents that don't contain pre-existing gene/protein entities are not processed
        """
        self.condition = condition
        if not os.path.exists(seth_fatjar_path):
            raise RuntimeError(f"required jar: {seth_fatjar_path} not found")
        self.gateway = JavaGateway.launch_gateway(
            classpath=seth_fatjar_path,
            die_on_exit=True,
            java_path=os.path.join(java_home, "bin", "java"),
        )
        self.seth = self.gateway.jvm.com.astrazeneca.kazu.SethRunner()
        self.entity_class = entity_class

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        if self.condition and not self.condition(doc):
            # skip this document
            return

        for section in doc.sections:
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
                        metadata={SETH_METADATA_KEY: python_dict},
                        mappings=[
                            Mapping(
                                default_label=self.entity_class,
                                source=self.entity_class,
                                parser_name="n/a",
                                idx=self.entity_class,
                                mapping_strategy=self.namespace(),
                                disambiguation_strategy=None,
                                confidence=LinkRanks.PROBABLE,
                            )
                        ],
                    ),
                )
            section.entities.extend(entities)
