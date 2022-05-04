import abc
from abc import abstractmethod
from typing import Tuple, List

from kazu.modelling.ontology_preprocessing.base import SynonymDatabase, StringNormalizer


class BlackLister(abc.ABC):
    """
    applies entity class specfic rules to a synonym, to see if it should be blacklisted or not
    """

    # def _collect_syn_set

    @abstractmethod
    def __call__(self, synonym: str) -> Tuple[bool, str]:
        raise NotImplementedError()


class DrugBlackLister:
    # CHEMBL drug names are often confused with genes and anatomy, for some reason
    def __init__(self, anatomy_synonym_sources: List[str], gene_synonym_sources: List[str]):
        self.syn_db = SynonymDatabase()
        self.gene_syns = set()
        for gene_synonym_source in gene_synonym_sources:
            self.gene_syns.update(set(self.syn_db.get_all(gene_synonym_source).keys()))
        self.anat_syns = set()
        for anat_synonym_source in anatomy_synonym_sources:
            self.anat_syns.update(set(self.syn_db.get_all(anat_synonym_source).keys()))

    def __call__(self, synonym: str) -> Tuple[bool, str]:
        norm = StringNormalizer.normalize(synonym)
        if norm in self.anat_syns:
            return False, "likely_anatomy"
        elif norm in self.gene_syns:
            return False, "likely_gene"
        elif len(synonym) <= 3 and not StringNormalizer.is_symbol_like(False, synonym):
            return False, "likely_bad_synonym"
        else:
            return True, "not_blacklisted"


class GeneBlackLister:
    # OT gene names are often confused with diseases,
    def __init__(self, disease_synonym_sources: List[str], gene_synonym_sources: List[str]):
        self.syn_db = SynonymDatabase()
        self.disease_syns = set()
        self.gene_syns = set()
        for disease_synonym_source in disease_synonym_sources:
            self.disease_syns.update(set(self.syn_db.get_all(disease_synonym_source).keys()))
        for gene_synonym_source in gene_synonym_sources:
            self.gene_syns.update(set(self.syn_db.get_all(gene_synonym_source).keys()))

    def __call__(self, synonym: str) -> Tuple[bool, str]:
        if synonym in self.gene_syns:
            return True, "not_blacklisted"
        elif StringNormalizer.normalize(synonym) in self.disease_syns:
            return False, "likely_disease"
        elif len(synonym) <= 3 and not StringNormalizer.is_symbol_like(False, synonym):
            return False, "likely_bad_synonym"
        else:
            return True, "not_blacklisted"


class DiseaseBlackLister:
    def __init__(self, disease_synonym_sources: List[str]):
        self.syn_db = SynonymDatabase()
        self.disease_syns = set()
        for disease_synonym_source in disease_synonym_sources:
            self.disease_syns.update(set(self.syn_db.get_all(disease_synonym_source).keys()))

    def __call__(self, synonym: str) -> Tuple[bool, str]:
        is_symbol_like = StringNormalizer.is_symbol_like(False, synonym)
        if synonym in self.disease_syns:
            return True, "not_blacklisted"
        elif is_symbol_like:
            return True, "not_blacklisted"
        elif len(synonym) <= 3 and not is_symbol_like:
            return False, "likely_bad_synonym"
        else:
            return True, "not_blacklisted"
