import os
import dataclasses
import logging
from typing import Optional, Callable
from collections.abc import Iterable

try:
    from py4j.java_gateway import JavaGateway
    from rdkit import Chem
except ImportError as e:
    raise ImportError(
        "To use OpsinStep, you need to install py4j and rdkit.\n"
        "You can either install these yourself, or install kazu[all-steps].\n"
    ) from e

from kazu.data import Document, Entity, CharSpan, Mapping, StringMatchConfidence
from kazu.steps import Step, document_iterating_step


BREAKS = set(" !@#&?|\t\n\r")  # https://www.acdlabs.com/iupac/nomenclature/93/r93_45.htm

logger = logging.getLogger(__name__)


class OpsinStep(Step):
    """A Step that calls Opsin (Open Parser for Systematic IUPAC Nomenclature) over
    py4j.

    :py:class:`~.TransformersModelForTokenClassificationNerStep` often identifies
    `IUPAC chemical nomenclature strings <https://en.wikipedia.org/wiki/IUPAC_nomenclature_of_organic_chemistry>`_
    as :class:`~.Entity`\\ s with an :attr:`~.Entity.entity_class` of ``drug``, but these entities
    fail to map to any of the drug parsers as no synonym is present. This step provides an extra
    way to resolve these chemical entities.

    `Opsin <https://opsin.ch.cam.ac.uk/>`_ produces a
    `SMILES <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system>`_
    from an IUPAC string and we use `rdkit <https://www.rdkit.org>`_ to convert that to a canonical
    SMILES to allow comparison between entities. This step then produces a :class:`~.Mapping` with
    the canonical SMILES string as the :attr:`~.Mapping.idx`\\ .

    Adding ``${OpsinStep}`` just after ``${MappingStep}`` in ``kazu/conf/Pipeline/default.yaml``
    will enable this step.

    .. attention::

        To use this step, you will need `py4j <https://www.py4j.org>`_ and
        `rdkit <https://www.rdkit.org>`_ installed, which is not installed as
        part of the default kazu install because this step isn't used as part
        of the default pipeline.

        You can either do:

        .. code-block:: console

            $ pip install py4j rdkit

        Or you can install required dependencies for all steps included in kazu
        with:

        .. code-block:: console

            $ pip install kazu[all-steps]

    .. note::

      The nature of this functionality is considered experimental and we may split it into two steps in the future, without
      making a major release. If you are using or are interested in using this step, please
      `open a GitHub issue <https://github.com/AstraZeneca/KAZU/issues/new>`_.

      .. raw:: html

        <details>
        <summary>Full details of possible change</summary>

      In particular, this step does two things:

      1. Adjust incorrect NER boundaries (particularly coming from the :class:`~.TransformersModelForTokenClassificationNerStep`\\ )
      2. Link drug entities consisting of IUPAC strings to a canonical SMILES

      The second of these aligns closely with the 'linking' stage in kazu, along with :class:`~.DictionaryEntityLinkingStep`\\ .
      We would ideally like to wrap the logic of 1. above into :class:`~.TransformersModelForTokenClassificationNerStep` like with the
      :class:`~.NonContiguousEntitySplitter` to fix these issues everywhere, and have 2. as a standalone linking step. However,
      this will require changes to the MappingLogic, and it may be tricky to de-couple 1 & 2 (the way this step currently does this
      depends on running Opsin to do accurately, which we would like to avoid doing twice, which may justify leaving this as a single
      step).

      .. raw:: html

        </details>

    .. testsetup::
        :skipif: kazu_model_pack_missing

        from pathlib import Path

        from hydra import initialize_config_dir, compose

        from kazu.utils.constants import HYDRA_VERSION_BASE

        # the hydra config is kept in the model pack
        cdir = Path(os.environ["KAZU_MODEL_PACK"]).joinpath("conf")

        with initialize_config_dir(version_base=HYDRA_VERSION_BASE, config_dir=str(cdir)):
            kazu_config = compose(config_name="config")

        from hydra.utils import instantiate

        from kazu.data import Mapping
        from kazu.steps.ner.opsin import OpsinStep

        opsin_step: OpsinStep = instantiate(kazu_config.OpsinStep)

        # function to make the doctests shorter
        def op(s: str) -> str:
            mapping: Optional[Mapping] = opsin_step.parseString(s)
            if mapping is None:
                return None
            else:
                return mapping.idx

    +------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Examples                                                                                                                                                                                             |
    +-----------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+
    | IUPAC Input                                                                                                                 | SMILES Output                                                          |
    +=============================================================================================================================+========================================================================+
    | Bicyclo[3.2.1]octane                                                                                                        | C1CC2CCC(C1)C2                                                         |
    |                                                                                                                             |                                                                        |
    |                                                                                                                             | .. doctest::                                                           |
    |                                                                                                                             |    :hide:                                                              |
    |                                                                                                                             |    :skipif: kazu_model_pack_missing                                    |
    |                                                                                                                             |                                                                        |
    |                                                                                                                             |    >>> op("Bicyclo[3.2.1]octane")                                      |
    |                                                                                                                             |    'C1CC2CCC(C1)C2'                                                    |
    +-----------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+
    | 2,2'-ethylenedipyridine                                                                                                     | c1ccc(CCc2ccccn2)nc1                                                   |
    |                                                                                                                             |                                                                        |
    |                                                                                                                             | .. doctest::                                                           |
    |                                                                                                                             |    :hide:                                                              |
    |                                                                                                                             |    :skipif: kazu_model_pack_missing                                    |
    |                                                                                                                             |                                                                        |
    |                                                                                                                             |    >>> op("2,2'-ethylenedipyridine")                                   |
    |                                                                                                                             |    'c1ccc(CCc2ccccn2)nc1'                                              |
    +-----------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+
    | Benzo[1",2":3,4;4",5":3',4']dicyclobuta[1,2-b:1',2'-c']difuran                                                              | c1cc2c3cc4c5cocc5c4cc3c2o1                                             |
    |                                                                                                                             |                                                                        |
    |                                                                                                                             | .. doctest::                                                           |
    |                                                                                                                             |    :hide:                                                              |
    |                                                                                                                             |    :skipif: kazu_model_pack_missing                                    |
    |                                                                                                                             |                                                                        |
    |                                                                                                                             |    >>> op(                                                             |
    |                                                                                                                             |    ...     "Benzo[1\\",2\\":3,4;4\\",5\\":3',4']dicyclobuta"               |
    |                                                                                                                             |    ...     "[1,2-b:1',2'-c']difuran"                                   |
    |                                                                                                                             |    ... )                                                               |
    |                                                                                                                             |    'c1cc2c3cc4c5cocc5c4cc3c2o1'                                        |
    +-----------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+
    | Cyclohexanone ethyl methyl ketal                                                                                            | CCOC1(OC)CCCCC1                                                        |
    |                                                                                                                             |                                                                        |
    |                                                                                                                             | .. doctest::                                                           |
    |                                                                                                                             |    :hide:                                                              |
    |                                                                                                                             |    :skipif: kazu_model_pack_missing                                    |
    |                                                                                                                             |                                                                        |
    |                                                                                                                             |    >>> op("Cyclohexanone ethyl methyl ketal")                          |
    |                                                                                                                             |    'CCOC1(OC)CCCCC1'                                                   |
    +-----------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+
    | 4-[2-(2-chloro-4-fluoroanilino)-5-methylpyrimidin-4-yl]-N-[(1S)-1-(3-chlorophenyl)-2-hydroxyethyl]-1H-pyrrole-2-carboxamide | Cc1cnc(Nc2ccc(F)cc2Cl)nc1-c1c[nH]c(C(=O)N[C@H](CO)c2cccc(Cl)c2)c1      |
    |                                                                                                                             |                                                                        |
    |                                                                                                                             | .. doctest::                                                           |
    |                                                                                                                             |    :hide:                                                              |
    |                                                                                                                             |    :skipif: kazu_model_pack_missing                                    |
    |                                                                                                                             |                                                                        |
    |                                                                                                                             |    >>> op(                                                             |
    |                                                                                                                             |    ...     "4-[2-(2-chloro-4-fluoroanilino)-5-methylpyrimidin"         |
    |                                                                                                                             |    ...     "-4-yl]-N-[(1S)-1-(3-chlorophenyl)-2-hydroxyethyl]"         |
    |                                                                                                                             |    ...     "-1H-pyrrole-2-carboxamide"                                 |
    |                                                                                                                             |    ... )                                                               |
    |                                                                                                                             |    'Cc1cnc(Nc2ccc(F)cc2Cl)nc1-c1c[nH]c(C(=O)N[C@H](CO)c2cccc(Cl)c2)c1' |
    +-----------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+
    | 7-cyclopentyl-5-(4-methoxyphenyl)pyrrolo[2,3-d]pyrimidin-4-amine                                                            | COc1ccc(-c2cn(C3CCCC3)c3ncnc(N)c23)cc1                                 |
    |                                                                                                                             |                                                                        |
    |                                                                                                                             | .. doctest::                                                           |
    |                                                                                                                             |    :hide:                                                              |
    |                                                                                                                             |    :skipif: kazu_model_pack_missing                                    |
    |                                                                                                                             |                                                                        |
    |                                                                                                                             |    >>> op(                                                             |
    |                                                                                                                             |    ...     "7-cyclopentyl-5-(4-methoxyphenyl)"                         |
    |                                                                                                                             |    ...     "pyrrolo[2,3-d]pyrimidin-4-amine"                           |
    |                                                                                                                             |    ... )                                                               |
    |                                                                                                                             |    'COc1ccc(-c2cn(C3CCCC3)c3ncnc(N)c23)cc1'                            |
    +-----------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+
    | | [(3S,3aS,6R,6aS)-3-nitrooxy-2,3,3a,5,6,6a-hexahydrofuro[3,2-b]furan-6-yl] nitrate                                         | O=[N+]([O-])O[C@H]1CO[C@H]2[C@@H]1OC[C@H]2O[N+](=O)[O-]                |
    | | (see `pubchem <https://pubchem.ncbi.nlm.nih.gov/compound/6883>`_)                                                         |                                                                        |
    |                                                                                                                             |                                                                        |
    |                                                                                                                             | .. doctest::                                                           |
    |                                                                                                                             |    :hide:                                                              |
    |                                                                                                                             |    :skipif: kazu_model_pack_missing                                    |
    |                                                                                                                             |                                                                        |
    |                                                                                                                             |    >>> op(                                                             |
    |                                                                                                                             |    ...     "[(3S,3aS,6R,6aS)-3-nitrooxy"                               |
    |                                                                                                                             |    ...     "-2,3,3a,5,6,6a-hexahydrofuro[3,2-b]furan-6-yl]"            |
    |                                                                                                                             |    ...     " nitrate"                                                  |
    |                                                                                                                             |    ... )                                                               |
    |                                                                                                                             |    'O=[N+]([O-])O[C@H]1CO[C@H]2[C@@H]1OC[C@H]2O[N+](=O)[O-]'           |
    +-----------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+
    | 1,4:3,6-dianhydro-2,5-di-O-Nitro-D-glucitol                                                                                 | | Opsin fails to parse this.                                           |
    |                                                                                                                             | | As a result, the Step will not produce a :class:`~.Mapping`\\ .       |
    |                                                                                                                             |                                                                        |
    |                                                                                                                             | .. doctest::                                                           |
    |                                                                                                                             |    :hide:                                                              |
    |                                                                                                                             |    :skipif: kazu_model_pack_missing                                    |
    |                                                                                                                             |                                                                        |
    |                                                                                                                             |    >>> op("1,4:3,6-dianhydro-2,5-di-O-Nitro-D-glucitol") is None       |
    |                                                                                                                             |    True                                                                |
    +-----------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+



    Paper:

    | Daniel M. Lowe, Peter T. Corbett, Peter Murray-Rust, and Robert C. Glen
    | Chemical Name to Structure: OPSIN, an Open Source Solution
    | Journal of Chemical Information and Modeling 2011 51 (3), 739-753
    | DOI: `10.1021/ci100384d <https://doi.org/10.1021/ci100384d>`_

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
        :param entity_class: search entities of this class for resolvable IUPAC string
        :param opsin_fatjar_path: path to a py4j fatjar, containing OPSIN dependencies
        :param java_home: path to installed java runtime
        :param condition: Since OPSIN can be slow, we can optionally specify a callable, so that
            any documents that don't contain pre-existing drug entities are not processed
        """
        self.condition = condition
        if not os.path.exists(opsin_fatjar_path):
            raise RuntimeError(f"required jar: {opsin_fatjar_path} not found")
        self.gateway = JavaGateway.launch_gateway(
            jarpath=".",
            classpath=opsin_fatjar_path,
            die_on_exit=True,
            java_path=os.path.join(java_home, "bin", "java"),
        )
        self.opsin = self.gateway.jvm.com.astrazeneca.kazu.OpsinRunner()
        self.entity_class = entity_class

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        if self.condition and not self.condition(doc):
            # skip this document
            return

        for section in doc.sections:
            updated_mappings = {}
            for ent in section.entities:
                if ent.entity_class == self.entity_class:
                    if len(ent.mappings) == 0:
                        # entity mapping failed, e.g. no exact matches to dictionaries
                        # look up to two spaces out
                        for (match, start, end) in self.extendString(ent, section.text, spaces=2):
                            maybe_mapping = self.parseString(match)
                            if maybe_mapping:
                                opsin_entity = dataclasses.replace(
                                    ent,
                                    match=match,
                                    spans=frozenset([CharSpan(start=start, end=end)]),
                                    mappings={maybe_mapping},
                                )
                                updated_mappings[ent] = opsin_entity
                                break

            for original_entity, opsin_entity in updated_mappings.items():
                section.entities.remove(original_entity)
                section.entities.append(opsin_entity)

    @staticmethod
    def extendString(ent: Entity, section: str, spaces: int = 0) -> Iterable[tuple[str, int, int]]:
        """Extend a possible IUPAC Entity match to longer plausible strings.

        :class:`~.TransformersModelForTokenClassificationNerStep` tends to truncate IUPAC matches to
        a first hyphen. Here, we attempt to extend the match to try to capture the full IUPAC string.

        :param ent: the entity we suspect contains (some of) an IUPAC string
        :param section: the section the entity is contained in
        :param spaces: the maximum number of IUPAC spaces/breaks to 'extend' the match through
        :return: plausible IUPAC strings with section start and end positions, ordered
            from longest to shortest.
        """
        start = ent.start
        end = ent.end
        res: list[tuple[str, int, int]] = []
        while start > 0 and section[start - 1] not in BREAKS:
            start = start - 1
        while end < len(section) and (section[end] not in BREAKS or spaces > 0):
            if section[end] in BREAKS:
                spaces = spaces - 1
                res.append((section[start:end], start, end))
            end = end + 1

        last_result = (section[start:end], start, end)
        if len(res) == 0 or not res[-1] == last_result:
            # res[-1] == last_result can be true when the while loop above
            # is broken out of immediately after encountering a BREAK and appending a result.
            # this can happen if there is a BREAK immediately before the end of
            # a section, or if there are two BREAKS in a row at the point at which
            # spaces decreases from 1 to 0
            res.append(last_result)
        # reversed because we prefer the longest strings
        yield from reversed(res)

    def parseString(self, name: str) -> Optional[Mapping]:
        """Attempt to parse a potential IUPAC drug name into a :class:`~.Mapping`\\ .

        Call Opsin with a potential IUPAC drug name to see if it parses - Opsin is fast enough
        that we can afford to try many potential candidates.

        If Opsin parses the string successfully, we convert the resulting
        `SMILES <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system>`_
        into a canonical SMILES using `rdkit <https://www.rdkit.org>`_\\ , and produce a
        :class:`~.Mapping` with the canonical SMILES as the :attr:`~.Mapping.idx`\\ .

        If Opsin parsing fails, ``None`` is returned. If the log level is set to debug,
        the error from Opsin for the parsing failure will be logged.

        :param name: the potential IUPAC drug name
        :return:
        """
        try:
            smiles = self.opsin.nameToStructure(name)
            if smiles is not None:
                smiles = Chem.CanonSmiles(smiles)
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
            logger.debug(f"Opsin parsing error: {reason}")
        return None
