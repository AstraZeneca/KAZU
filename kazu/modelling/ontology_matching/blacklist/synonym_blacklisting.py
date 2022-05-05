import abc
from abc import abstractmethod
from typing import Tuple, List, Dict, Optional
import pandas as pd

from kazu.modelling.ontology_preprocessing.base import SynonymDatabase, StringNormalizer


class AnnotationLookup:
    def __init__(self, annotations_path: str):
        self.annotations = self.df_to_dict(pd.read_csv(annotations_path))

    def df_to_dict(self, df: pd.DataFrame) -> Dict[str, Dict]:
        return df.set_index("match").to_dict(orient="index")

    def __call__(self, synonym: str) -> Optional[Tuple[bool, str]]:
        annotation_info = self.annotations.get(synonym)
        if annotation_info:
            action = annotation_info["action"]
            if action == "keep":
                return True, "annotated_keep"
            elif action == "drop":
                return False, "annotated_drop"
            else:
                raise ValueError(f"{action} is not valid")
        else:
            return None


class BlackLister(abc.ABC):
    """
    applies entity class specfic rules to a synonym, to see if it should be blacklisted or not
    """

    # def _collect_syn_set

    @abstractmethod
    def __call__(self, synonym: str) -> Tuple[bool, str]:
        """

        :param synonym: synonym to test
        :return: tuple of whether synoym is good True|False, and the reason for the decision
        """
        raise NotImplementedError()


class DrugBlackLister:
    # CHEMBL drug names are often confused with genes and anatomy, for some reason
    def __init__(
        self,
        annotation_lookup: AnnotationLookup,
        anatomy_synonym_sources: List[str],
        gene_synonym_sources: List[str],
    ):
        self.annotation_lookup = annotation_lookup
        self.syn_db = SynonymDatabase()
        self.gene_syns = set()
        for gene_synonym_source in gene_synonym_sources:
            self.gene_syns.update(set(self.syn_db.get_all(gene_synonym_source).keys()))
        self.anat_syns = set()
        for anat_synonym_source in anatomy_synonym_sources:
            self.anat_syns.update(set(self.syn_db.get_all(anat_synonym_source).keys()))

    def __call__(self, synonym: str) -> Tuple[bool, str]:
        lookup_result = self.annotation_lookup(synonym)
        if lookup_result:
            return lookup_result
        else:
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
    def __init__(
        self,
        annotation_lookup: AnnotationLookup,
        disease_synonym_sources: List[str],
        gene_synonym_sources: List[str],
    ):
        self.annotation_lookup = annotation_lookup
        self.syn_db = SynonymDatabase()
        self.disease_syns = set()
        self.gene_syns = set()
        for disease_synonym_source in disease_synonym_sources:
            self.disease_syns.update(set(self.syn_db.get_all(disease_synonym_source).keys()))
        for gene_synonym_source in gene_synonym_sources:
            self.gene_syns.update(set(self.syn_db.get_all(gene_synonym_source).keys()))

    def __call__(self, synonym: str) -> Tuple[bool, str]:
        lookup_result = self.annotation_lookup(synonym)
        if lookup_result:
            return lookup_result
        else:
            if synonym in self.gene_syns:
                return True, "not_blacklisted"
            elif StringNormalizer.normalize(synonym) in self.disease_syns:
                return False, "likely_disease"
            elif len(synonym) <= 3 and not StringNormalizer.is_symbol_like(False, synonym):
                return False, "likely_bad_synonym"
            else:
                return True, "not_blacklisted"


class DiseaseBlackLister:
    def __init__(self, annotation_lookup: AnnotationLookup, disease_synonym_sources: List[str]):
        self.annotation_lookup = annotation_lookup
        self.syn_db = SynonymDatabase()
        self.disease_syns = set()
        for disease_synonym_source in disease_synonym_sources:
            self.disease_syns.update(set(self.syn_db.get_all(disease_synonym_source).keys()))

    def __call__(self, synonym: str) -> Tuple[bool, str]:
        lookup_result = self.annotation_lookup(synonym)
        if lookup_result:
            return lookup_result
        else:

            is_symbol_like = StringNormalizer.is_symbol_like(False, synonym)
            if synonym in self.disease_syns:
                return True, "not_blacklisted"
            elif is_symbol_like:
                return True, "not_blacklisted"
            elif len(synonym) <= 3 and not is_symbol_like:
                return False, "likely_bad_synonym"
            else:
                return True, "not_blacklisted"
