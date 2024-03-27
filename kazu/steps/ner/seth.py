import os
from typing import Optional
from collections.abc import Callable

try:
    from py4j.java_gateway import JavaGateway
except ImportError as e:
    raise ImportError(
        "To use SethStep, you need to install py4j.\n"
        "You can either install py4j yourself, or install kazu[all-steps].\n"
    ) from e

from kazu.data import Document, Entity, Mapping, StringMatchConfidence
from kazu.steps import Step, document_iterating_step


SETH_METADATA_KEY = "seth"


class SethStep(Step):
    """A Step that calls SETH (SNP Extraction Tool for Human Variations) over py4j.

    .. attention::

        To use this step, you will need `py4j <https://www.py4j.org>`_ installed,
        which is not installed as part of the default kazu install because this
        step isn't used as part of the default pipeline.

        You can either do:

        .. code-block:: console

            $ pip install py4j

        Or you can install required dependencies for all steps included in kazu
        with:

        .. code-block:: console

            $ pip install kazu[all-steps]

    Paper:

    | Thomas, P., Rocktäschel, T., Hakenberg, J., Mayer, L., and Leser, U. (2016).
    | `SETH detects and normalizes genetic variants in text. <https://pubmed.ncbi.nlm.nih.gov/27256315/>`_
    | Bioinformatics (2016)

    .. raw:: html

        <details>
        <summary>Bibtex Citation Details</summary>

    .. code:: bibtex

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

    .. raw:: html

        </details>
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
        :param java_home: path to installed java runtime
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
            mutation_lst = self.seth.findMutations(section.text)
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
                        text=section.text,
                        entity_class=self.entity_class,
                        namespace=self.namespace(),
                        metadata={SETH_METADATA_KEY: python_dict},
                        mappings=[
                            Mapping(
                                default_label=self.entity_class,
                                source=self.entity_class,
                                parser_name="n/a",
                                idx=self.entity_class,
                                string_match_strategy=self.namespace(),
                                disambiguation_strategy=None,
                                string_match_confidence=StringMatchConfidence.PROBABLE,
                            )
                        ],
                    ),
                )
            section.entities.extend(entities)
