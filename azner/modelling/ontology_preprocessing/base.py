import itertools
import json
import re
from pathlib import Path
from typing import List

import pandas as pd
import rdflib
from rdflib import URIRef
import sqlite3
from tqdm.auto import tqdm
from azner.utils.link_index import DEFAULT_LABEL, IDX, SYN, MAPPING_TYPE, SOURCE

from abc import ABC
import spacy


class StopWordRemover:
    """
    remove stopwords from a string
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.all_stopwords = self.nlp.Defaults.stop_words

    def __call__(self, text: str) -> str:
        lst = []
        for token in text.split():
            if token.lower() not in self.all_stopwords:  # checking whether the word is not
                lst.append(token)
        return " ".join(lst)


class OntologyParser(ABC):
    """
    Parse an ontology (or similar) into a set of outputs suitable for NLP entity linking
    Implementations should have a class attribute 'name' to something suitably representative
    """

    name = "unnamed"
    training_col_names = ["id", "syn1", "syn2"]
    minimum_default_label_column_names = [SOURCE, DEFAULT_LABEL, IDX]
    all_synonym_column_names = [IDX, DEFAULT_LABEL, SYN, MAPPING_TYPE]

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
        # ensure correct order
        df = df[self.all_synonym_column_names]
        # make sure default labels are also in the synonym list
        default_labels_df = df[[IDX, DEFAULT_LABEL]].drop_duplicates()
        default_labels_df[SYN] = default_labels_df[DEFAULT_LABEL]
        df = pd.concat([df, default_labels_df])
        df = df.dropna(axis=0)
        df = df.drop_duplicates()
        df.sort_values(by=[IDX, DEFAULT_LABEL, SYN], inplace=True)
        return df

    def format_synonym_table(self) -> pd.DataFrame:
        """
        implementations should override this method, returning a 'long, thin' pd.DataFrame of
        ["id", "default_label", SYN,"mapping_type"]
        id: the ontology id
        default_label: the preferred label
        syn: a synonym of the concept
        mapping_type: the type of mapping from default label to synonym - e.g. xref, exactSyn etc. Usually defined by
                    the ontology
        :return:
        """
        raise NotImplementedError()

    def format_default_labels(self) -> pd.DataFrame:
        """
        get a dataframe of default labels and ids. Useful for generating e.g. embeddings
        :return:
        """
        self.cache_synonym_table()
        default_label_df = self.synonym_table[[IDX, DEFAULT_LABEL]].drop_duplicates().copy()
        default_label_df[SOURCE] = self.name
        return default_label_df

    def format_training_table(self) -> pd.DataFrame:
        """
        generate a table of synonym pairs. Useful for aligning an embedding space (e.g. as for sapbert)
        :return:
        """
        self.cache_synonym_table()
        tqdm.pandas(desc=f"generating training pairs for {self.name}")
        df = self.synonym_table.groupby(by=[IDX]).progress_apply(self.select_pos_pairs)
        df.index = [i for i in range(df.shape[0])]
        df = df[[0, 1, IDX]]
        df.columns = OntologyParser.training_col_names
        df[IDX] = df[IDX].astype("category").cat.codes
        return df

    def select_pos_pairs(self, df: pd.Series):
        """
        select synonym pair combinations for alignment. Capped at 50 to prevent overfitting
        :param df:
        :return:
        """
        id = df[IDX].unique()[0]
        labels = df[SYN].unique()
        if len(labels) > 50:
            labels = list(labels)[:50]
        combinations = list(itertools.combinations(labels, 2))
        new_df = pd.DataFrame(combinations)
        new_df[IDX] = id
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
        df = pd.DataFrame.from_dict({DEFAULT_LABEL: default_labels, IDX: iris, SYN: syns})
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

    def __init__(self, in_path: str):
        super().__init__(in_path)
        self.sw_remover = StopWordRemover()

    def format_synonym_table(self) -> pd.DataFrame:
        x = json.load(open(self.in_path, "r"))
        graph = x["graphs"][0]
        nodes = graph["nodes"]
        ids = []
        default_label = []
        all_syns = []
        mapping_type = []
        for i, node in enumerate(nodes):
            syns = node.get("meta", {}).get("synonyms", [])
            for syn_dict in syns:
                pred = syn_dict["pred"]
                mapping_type.append(pred)
                syn = syn_dict["val"]
                ids.append(node["id"])
                default_label.append(node.get("lbl"))
                all_syns.append(syn)
                no_stops_syn = self.sw_remover(syn)
                if no_stops_syn != syn:
                    ids.append(node["id"])
                    default_label.append(node.get("lbl"))
                    all_syns.append(no_stops_syn)
                    mapping_type.append(pred)

        df = pd.DataFrame.from_dict(
            {IDX: ids, DEFAULT_LABEL: default_label, SYN: all_syns, MAPPING_TYPE: mapping_type}
        )
        return df


class EnsemblOntologyParser(OntologyParser):
    name = "ENSEMBL"
    """
    input is a json from HGNC
     e.g. http://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/json/hgnc_complete_set.json
    :return:
    """

    GREEK_SUBS = {
        "\u0391": "alpha",
        "\u0392": "beta",
        "\u0393": "gamma",
        "\u0394": "delta",
        "\u0395": "epsilon",
        "\u0396": "zeta",
        "\u0397": "eta",
        "\u0398": "theta",
        "\u0399": "iota",
        "\u039A": "kappa",
        "\u039B": "lamda",
        "\u039C": "mu",
        "\u039D": "nu",
        "\u039E": "xi",
        "\u039F": "omicron",
        "\u03A0": "pi",
        "\u03A1": "rho",
        "\u03A3": "sigma",
        "\u03A4": "tau",
        "\u03A5": "upsilon",
        "\u03A6": "phi",
        "\u03A7": "chi",
        "\u03A8": "psi",
        "\u03A9": "omega",
        "\u03F4": "theta",
        "\u03B1": "alpha",
        "\u03B2": "beta",
        "\u03B3": "gamma",
        "\u03B4": "delta",
        "\u03B5": "epsilon",
        "\u03B6": "zeta",
        "\u03B7": "eta",
        "\u03B8": "theta",
        "\u03B9": "iota",
        "\u03BA": "kappa",
        "\u03BC": "mu",
        "\u03BD": "nu",
        "\u03BE": "xi",
        "\u03BF": "omicron",
        "\u03C0": "pi",
        "\u03C1": "rho",
        "\u03C2": "final sigma",
        "\u03C3": "sigma",
        "\u03C4": "tau",
        "\u03C5": "upsilon",
        "\u03C6": "phi",
        "\u03C7": "chi",
        "\u03C8": "psi",
        "\u03C9": "omega",
    }

    GREEK_SUBS_ABBRV = {k: v[0] for k, v in GREEK_SUBS.items()}
    GREEK_SUBS_REVERSED = {v: k for k, v in GREEK_SUBS.items()}

    EXCLUDED_PARENTHESIS = ["", "non-protein coding"]

    def format_synonym_table(self) -> pd.DataFrame:

        keys_to_check = [
            "name",
            "symbol",
            "uniprot_ids",
            "alias_name",
            "alias_symbol",
            "prev_name",
            "lncipedia",
            "prev_symbol",
            "vega_id",
            "refseq_accession",
            "hgnc_id",
            "mgd_id",
            "rgd_id",
            "ccds_id",
            "pseudogene.org",
        ]

        with open(self.in_path, "r") as f:
            data = json.load(f)
        ids = []
        default_label = []
        all_syns = []

        docs = data["response"]["docs"]
        for doc in docs:

            def get_with_default_list(key: str):
                found = doc.get(key, [])
                if not isinstance(found, list):
                    found = [found]
                return found

            ensembl_gene_id = doc.get("ensembl_gene_id", None)
            name = doc.get("name", None)
            if ensembl_gene_id is None or name is None:
                continue
            else:
                # find synonyms
                synonyms = []
                for x in keys_to_check:
                    synonyms_this_entity = get_with_default_list(x)
                    for y in synonyms_this_entity:
                        synonyms.extend(self.post_process_synonym(y))

                synonyms = list(set(synonyms))
                # filter any very short matches
                synonyms = [x for x in synonyms if len(x) > 2]
                [ids.append(ensembl_gene_id) for _ in range(len(synonyms))]
                [default_label.append(name) for _ in range(len(synonyms))]
                all_syns.extend(synonyms)

        df = pd.DataFrame.from_dict({IDX: ids, DEFAULT_LABEL: default_label, SYN: all_syns})
        return df

    def post_process_synonym(self, syn: str) -> List[str]:
        """
        need to also do some basic string processing on HGNC
        :param syn:
        :return:
        """
        to_add = []
        paren_re = r"(.*)\((.*)\)(.*)"
        to_add.append(syn)
        if "(" in syn and ")" in syn:
            # expand brackets
            matches = re.match(paren_re, syn)
            if matches is not None:
                all_groups_no_brackets = []
                for group in matches.groups():
                    if group not in self.EXCLUDED_PARENTHESIS:
                        to_add.append(group)
                        all_groups_no_brackets.append(group)
                to_add.append("".join(all_groups_no_brackets))
        # expand slashes
        for x in range(len(to_add)):
            if "/" in to_add[x]:
                splits = to_add[x].split("/")
                to_add.extend(splits)

        # sub greek
        for x in range(len(to_add)):
            to_add.append(self.substitute_greek_unicode(to_add[x]))
            to_add.append(self.substitute_english_with_greek_unicode(to_add[x]))
            to_add.append(self.substitute_greek_unicode_abbrvs(to_add[x]))

        return to_add

    def substitute_greek_unicode(self, text: str) -> str:
        if any([x in text for x in self.GREEK_SUBS.keys()]):
            for greek_unicode in self.GREEK_SUBS.keys():
                if greek_unicode in text:
                    text = text.replace(greek_unicode, self.GREEK_SUBS[greek_unicode])
                    text = self.substitute_greek_unicode(text)
            return text
        else:
            return text

    def substitute_greek_unicode_abbrvs(self, text: str) -> str:
        if any([x in text for x in self.GREEK_SUBS_ABBRV.keys()]):
            for greek_unicode in self.GREEK_SUBS_ABBRV.keys():
                if greek_unicode in text:
                    text = text.replace(greek_unicode, self.GREEK_SUBS_ABBRV[greek_unicode])
                    text = self.substitute_greek_unicode_abbrvs(text)
            return text
        else:
            return text

    def substitute_english_with_greek_unicode(self, text: str) -> str:
        if any([x in text for x in self.GREEK_SUBS_REVERSED.keys()]):
            for greek_unicode in self.GREEK_SUBS_REVERSED.keys():
                if greek_unicode in text:
                    text = text.replace(greek_unicode, self.GREEK_SUBS_REVERSED[greek_unicode])
                    text = self.substitute_english_with_greek_unicode(text)
            return text
        else:
            return text


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
        df[DEFAULT_LABEL] = df[DEFAULT_LABEL].str.lower()
        df[SYN] = df[SYN].str.lower()

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
        df = pd.DataFrame.from_dict({IDX: ids, DEFAULT_LABEL: default_labels, SYN: all_syns})
        return df
