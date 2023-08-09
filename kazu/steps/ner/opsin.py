import os
from typing import Optional, Callable

from py4j.java_gateway import JavaGateway

from rdkit import Chem

from kazu.data.data import Document, Entity, Mapping, StringMatchConfidence
from kazu.steps import Step, document_iterating_step


OPSIN_METADATA_KEY = "opsin"
BREAKS = " !@#&?|\t\n\r" # https://www.acdlabs.com/iupac/nomenclature/93/r93_45.htm


class OpsinStep(Step):
    """A Step that calls Opsin (Open Parser for Systematic IUPAC Nomenclature) over py4j.

    :py:class:`~.TransformersModelForTokenClassificationNerStep` often identifies IUPAC strings as entity_class=drug,
    but they fail to map to one of the drug ontology dictionaries. This service provides an extra way to resolve chemical entities.
    Opsin produces a SMILES from an IUPAC string and we use rdkit to convert that to a canonical SMILES for comparison, as an IDX.

    Adding ${OpsinStep} just after ${MappingStep} in kazu/conf/Pipeline/default.yaml will enable this step.

    A test: BREXPIPRAZOLE is great and is the same as OPC-34712 but not Bicyclo[3.2.1]octane or 2,2'-ethylenedipyridine or Benzo[1\",2\":3,4;4\",5\":3',4']dicyclobuta[1,2-b:1',2'-c']difuran or Cyclohexanone ethyl methyl ketal or 4-[2-(2-chloro-4-fluoroanilino)-5-methylpyrimidin-4-yl]-N-[(1S)-1-(3-chlorophenyl)-2-hydroxyethyl]-1H-pyrrole-2-carboxamide added to 7-cyclopentyl-5-(4-methoxyphenyl)pyrrolo[2,3-d]pyrimidin-4-amine

    Examples:
        Bicyclo[3.2.1]octane
        2,2'-ethylenedipyridine
        Benzo[1",2":3,4;4",5":3',4']dicyclobuta[1,2-b:1',2'-c']difuran
        Cyclohexanone ethyl methyl ketal
        4-[2-(2-chloro-4-fluoroanilino)-5-methylpyrimidin-4-yl]-N-[(1S)-1-(3-chlorophenyl)-2-hydroxyethyl]-1H-pyrrole-2-carboxamide
        7-cyclopentyl-5-(4-methoxyphenyl)pyrrolo[2,3-d]pyrimidin-4-amine

    Tough cases:
        Fails to parse 1,4:3,6-dianhydro-2,5-di-O-Nitro-D-glucitol but does parse [(3S,3aS,6R,6aS)-3-nitrooxy-2,3,3a,5,6,6a-hexahydrofuro[3,2-b]furan-6-yl] nitrate  see https://pubchem.ncbi.nlm.nih.gov/compound/6883


    Paper:

    Daniel M. Lowe, Peter T. Corbett, Peter Murray-Rust, and Robert C. Glen
    Chemical Name to Structure: OPSIN, an Open Source Solution
    Journal of Chemical Information and Modeling 2011 51 (3), 739-753
    DOI: [10.1021/ci100384d](https://doi.org/10.1021/ci100384d)

    .. raw:: html

        <details>
        <summary>Bibtex Citation Details</summary>

    .. code:: bibtex

        @article{doi:10.1021/ci100384d,
        author = {Lowe, Daniel M. and Corbett, Peter T. and Murray-Rust, Peter and Glen, Robert C.},
        title = {Chemical Name to Structure: OPSIN, an Open Source Solution},
        journal = {Journal of Chemical Information and Modeling},
        volume = {51},
        number = {3},
        pages = {739-753},
        year = {2011},
        doi = {10.1021/ci100384d},
            note ={PMID: 21384929},

        URL = { 
                https://doi.org/10.1021/ci100384d
            
        },
        eprint = { 
                https://doi.org/10.1021/ci100384d
        }
        }

    .. raw:: html

        </details>

    """

    def __init__(
        self,
        entity_class: str,
        opsin_fatjar_path: str,
        java_home: str,
        condition: Optional[Callable[[Document], bool]] = None,
    ):
        """
        :param entity_class: the entity_class to assign to any Entities that emerge
        :param opsin_fatjar_path: path to a py4j fatjar, containing OPSIN dependencies
        :param condition: Since OPSIN can be slow, we can optionally specify a callable, so that
            any documents that don't contain pre-existing drug entities are not processed
        """
        self.condition = condition
        if not os.path.exists(opsin_fatjar_path):
            raise RuntimeError(f"required jar: {opsin_fatjar_path} not found")
        self.gateway = JavaGateway.launch_gateway(
            jarpath='.',
            classpath=opsin_fatjar_path,
            die_on_exit=True,
            java_path=os.path.join(java_home, "bin", "java"),
        )
        self.opsin = self.gateway.jvm.com.astrazeneca.kazu.OpsinRunner()
        self.entity_class = entity_class

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        for section in doc.sections:
            for ent in section.entities:
                if ent.entity_class == self.entity_class: # entity is a drug
                    if len(ent.mappings) == 0: # entity mapping failed, e.g., no exact matches to dictionaries
                        mapping = None
                        for spaces in range(2,-1,-1): # look up to two spaces out
                            if mapping == None:
                                testStr, ridx, fidx = self.extendString(ent, section.get_text(), spaces)
                                mapping = self.parseString(testStr)
                                if mapping != None and testStr != ent.match: # update entity match to expanded string
                                    ent.match = testStr
                                    ent.match_norm = testStr
                                    ent.start = ridx
                                    ent.end = fidx
                        if mapping != None:
                            ent.mappings=[mapping]
                            ent.syn_term_to_synonym_terms = dict() # remove close synonym matches

    # TransformersModelForTokenClassificationNerStep tends to truncate the IUPAC match to a first hyphen
    # Here we extend the entity match
    @staticmethod
    def extendString(ent: Entity, section: str, spaces: int = 0):
        ridx = ent.start
        fidx = ent.end
        if ent.match != section[ent.start:ent.end]:
            ridx = section.upper().find(ent.match.upper())
            if ridx > -1:
                fidx = ridx + len(ent.match)
            else:
                ridx = ent.start # we might need to try harder to find the index into the original text string, but punt at the moment
        while ridx > 0 and section[ridx-1] not in BREAKS:
            ridx = ridx - 1
        while fidx < len(section) and (section[fidx] not in BREAKS or spaces > 0):
            if section[fidx] in BREAKS:
                spaces = spaces - 1
            fidx = fidx + 1
        entStr = section[ridx:fidx]
        return entStr, ridx, fidx

    def parseString(self, name: str) -> Mapping:
        try:
            smiles = self.opsin.nameToStructure(name)
            if smiles != None:
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                mapping = Mapping(
                                default_label=name,
                                source="Opsin",
                                parser_name="Opsin",
                                idx=smiles,
                                string_match_strategy=self.namespace(),
                                disambiguation_strategy=None,
                                string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                            )
                return mapping
        except Exception as e:
            reason = e.args[1].getMessage()
            #print("Opsin parsing error:"+str(reason))
        return None
