import itertools
import json
from pathlib import Path
from typing import List

import pandas as pd
import rdflib
from rdflib import URIRef
import sqlite3
from tqdm.auto import tqdm


class OntologyParser:
    """
    Parse an ontology (or similar) into a set of outputs suitable for NLP entity linking
    Implementations should override static field 'name' to something suitably representative
    """

    name = "unnamed"
    training_col_names = ["id", "syn1", "syn2"]
    default_label_column_names = ["source", "default_label", "iri"]
    all_synonym_column_names = ["iri", "default_label", "syn"]

    def __init__(self, in_path: str):
        """

        :param in_path: Path to some resource that should be processed (e.g. owl file, db config, tsv etc)
        """
        self.in_path = in_path
        self.synonym_table = None

    def cache_synonym_table(self):
        """
        populate self.synonym_table, if not already done so
        :return:
        """
        if self.synonym_table is None:
            self.synonym_table = self.post_process_synonym_table()
        assert len(
            set(self.synonym_table.columns) & set(OntologyParser.all_synonym_column_names)
        ) == len(OntologyParser.all_synonym_column_names)

    def post_process_synonym_table(self) -> pd.DataFrame:
        df = self.format_synonym_table()
        df.columns = self.all_synonym_column_names
        # make sure default labels are also in the synonym list
        default_labels_df = df[["iri", "default_label"]].drop_duplicates()
        default_labels_df["syn"] = default_labels_df["default_label"]
        df = pd.concat([df, default_labels_df])
        df = df.dropna(axis=0)
        df = df.drop_duplicates()
        df.sort_values(by=["iri", "default_label", "syn"], inplace=True)
        return df

    def format_synonym_table(self) -> pd.DataFrame:
        """
        implementations should override this method, returning a 'long, thin' pd.DataFrame of
        ["id", "default_label", "syn"]
        id: the ontology id
        default_label: the preferred label
        syn: a synonym of the concept
        :return:
        """
        raise NotImplementedError()

    def format_default_labels(self) -> pd.DataFrame:
        """
        get a dataframe of default labels and ids. Useful for generating e.g. embeddings
        :return:
        """
        self.cache_synonym_table()
        default_label_df = self.synonym_table[["iri", "default_label"]].drop_duplicates().copy()
        default_label_df["source"] = self.name
        return default_label_df

    def format_training_table(self) -> pd.DataFrame:
        """
        generate a table of synonym pairs. Useful for aligning an embedding space (e.g. as for sapbert)
        :return:
        """
        self.cache_synonym_table()
        tqdm.pandas(desc=f"generating training pairs for {self.name}")
        df = self.synonym_table.groupby(by=["iri"]).progress_apply(self.select_pos_pairs)
        df.index = [i for i in range(df.shape[0])]
        df = df[[0, 1, "iri"]]
        df.columns = OntologyParser.training_col_names
        df["id"] = df["id"].astype("category").cat.codes
        return df

    def select_pos_pairs(self, df: pd.Series):
        """
        select synonym pair combinations for alignment. Capped at 50 to prevent overfitting
        :param df:
        :return:
        """
        id = df["iri"].unique()[0]
        labels = df["syn"].unique()
        if len(labels) > 50:
            labels = list(labels)[:50]
        combinations = list(itertools.combinations(labels, 2))
        new_df = pd.DataFrame(combinations)
        new_df["iri"] = id
        return new_df

    def write_training_pairs(self, out_path: str):
        """
        write training pairs to a directory.
        :param out_path: directory to write to
        :return:
        """
        path = Path(out_path)
        if not path.is_dir():
            raise RuntimeError(f"{path} is not a directory")
        self.format_training_table().to_parquet(
            path.joinpath(f"{self.name}_training_pairs.parquet"), index=None
        )

    def write_synonym_table(self, out_path: str):
        """
        write synonym table to a directory.
        :param out_path: directory to write to
        :return:
        """
        self.cache_synonym_table()
        path = Path(out_path)
        if not path.is_dir():
            raise RuntimeError(f"{path} is not a directory")
        self.synonym_table.to_parquet(path.joinpath(f"{self.name}_synonyms.parquet"), index=None)

    def write_default_labels(self, out_path: str):
        """
        write default labels to a directory.
        :param out_path: directory to write to
        :return:
        """
        path = Path(out_path)
        if not path.is_dir():
            raise RuntimeError(f"{path} is not a directory")
        self.format_default_labels().to_parquet(
            path.joinpath(f"{self.name}_default_labels.parquet"), index=None
        )


class RDFGraphParser(OntologyParser):
    """
    Parser for Owl files.
    """

    name = "RDFGraphParser"

    def _get_synonym_predicates(self) -> List[str]:
        """
        subclasses should override this. Returns a List[str] of rdf predicates used to select synonyms from the owl
        graph
        :return:
        """
        raise NotImplementedError()

    def format_synonym_table(self) -> pd.DataFrame:
        g = rdflib.Graph()
        g.parse(self.in_path)
        label_predicates = URIRef("http://www.w3.org/2000/01/rdf-schema#label")
        synonym_predicates = [URIRef(x) for x in self._get_synonym_predicates()]
        default_labels = []
        iris = []
        syns = []

        for sub, obj in g.subject_objects(label_predicates):
            default_labels.append(str(obj))
            iris.append(str(sub))
            syns.append(str(obj))
            for syn_predicate in synonym_predicates:
                for other_syn_obj in g.objects(subject=sub, predicate=syn_predicate):
                    default_labels.append(str(obj))
                    iris.append(str(sub))
                    syns.append(str(other_syn_obj))
        df = pd.DataFrame.from_dict({"default_label": default_labels, "iri": iris, "syn": syns})
        return df


class UberonParser(RDFGraphParser):
    name = "UBERON"
    """
    input should be an UBERON owl file
    e.g.
    https://www.ebi.ac.uk/ols/ontologies/uberon
    """

    def _get_synonym_predicates(self) -> List[str]:
        return [
            "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym",
            "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
        ]


class MondoParser(OntologyParser):
    name = "MONDO"
    """
    input should be an MONDO owl file
    e.g.
    https://www.ebi.ac.uk/ols/ontologies/mondo
    """

    def format_synonym_table(self) -> pd.DataFrame:
        x = json.load(open(self.in_path, "r"))
        graph = x["graphs"][0]
        nodes = graph["nodes"]
        ids = []
        default_label = []
        all_syns = []
        for i, node in enumerate(nodes):
            syns = node.get("meta", {}).get("synonyms", [])
            for syn_dict in syns:
                if syn_dict["pred"] == "hasExactSynonym":
                    ids.append(node["id"])
                    default_label.append(node.get("lbl"))
                    all_syns.append(syn_dict["val"])
        df = pd.DataFrame.from_dict({"iri": ids, "default_label": default_label, "syn": all_syns})
        return df


class EnsemblOntologyParser(OntologyParser):
    name = "ENSEMBL"
    """
    input is a tsv file with three cols, that line up to ["iri", "default_label", "syn"]
    Typically exported from Ensembl BioMart https://www.ensembl.org/biomart/
    :return:
    """

    def format_synonym_table(self) -> pd.DataFrame:
        df = pd.read_csv(self.in_path, sep="\t")
        return df


class ChemblOntologyParser(OntologyParser):
    name = "CHEMBL"
    """
    input is a sqllite dump from Chembl, e.g.
    https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_29_sqlite.tar.gz
    :return:
    """

    def format_synonym_table(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.in_path)
        query = """
            SELECT chembl_id AS iri, pref_name AS default_label, synonyms AS syn
            FROM molecule_dictionary AS md
                     JOIN molecule_synonyms ms ON md.molregno = ms.molregno
            UNION ALL
            SELECT chembl_id AS iri, pref_name AS default_label, pref_name AS syn
            FROM molecule_dictionary
        """
        df = pd.read_sql(query, conn)
        df["default_label"] = df["default_label"].str.lower()
        df["syn"] = df["syn"].str.lower()

        df.drop_duplicates(inplace=True)

        return df


class CLOOntologyParser(RDFGraphParser):
    name = "CLO"
    """
    input is a CLO Owl file
    https://www.ebi.ac.uk/ols/ontologies/clo
    """

    def _get_synonym_predicates(self) -> List[str]:
        return [
            "http://purl.obolibrary.org/obo/hasNarrowSynonym",
            "http://purl.obolibrary.org/obo/hasExactSynonym",
            "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym",
        ]


class CellosaurusOntologyParser(OntologyParser):
    name = "CELLOSAURUS"
    """
    input is an obo file from cellosaurus, e.g.
    https://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo
    :return:
    """

    def format_synonym_table(self) -> pd.DataFrame:

        ids = []
        default_labels = []
        all_syns = []
        with open(self.in_path, "r") as f:
            id = None
            default_label = None
            for line in f:
                text = line.rstrip()
                if text.startswith("id:"):
                    id = text.split(" ")[1]
                elif text.startswith("name:"):
                    default_label = text.split(" ")[1][1:]
                elif text.startswith("synonym:"):
                    syn = text.split(" ")[1][1:-1]
                    ids.append(id)
                    default_labels.append(default_label)
                    all_syns.append(syn)
                else:
                    pass
        df = pd.DataFrame.from_dict({"iri": ids, "default_label": default_labels, "syn": all_syns})
        return df
