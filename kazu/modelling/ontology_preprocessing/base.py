import copy
import itertools
import json
import logging
import os
import sqlite3
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterable, Optional, DefaultDict

import cachetools
import pandas as pd
import rdflib
from rdflib import URIRef
from tqdm.auto import tqdm

# dataframe column keys
from kazu.modelling.ontology_preprocessing.synonym_generation import SynonymData, SynonymGenerator

DEFAULT_LABEL = "default_label"
IDX = "idx"
SYN = "syn"
MAPPING_TYPE = "mapping_type"
SOURCE = "source"

logger = logging.getLogger(__name__)


class MetadataDatabase:
    """
    Singleton of Ontology metadata database. Purpose: metadata needs to be looked up in different linking processes,
    and this singleton allows us to load it once/reduce memory usage
    """

    instance: Optional["__MetadataDatabase"] = None

    class __MetadataDatabase:
        database: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)

        def add(self, name: str, metadata: Dict[str, Any]):
            self.database[name].update(metadata)

        def get(self, name: str, idx: str) -> Any:
            return self.database[name].get(idx)

    def __init__(self):
        if not MetadataDatabase.instance:
            MetadataDatabase.instance = MetadataDatabase.__MetadataDatabase()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def get(self, name: str, idx: str) -> Any:
        """
        get the metadata associated with an ontology and id
        :param name: name of ontology to query
        :param idx: idx to query
        :return:
        """
        return copy.deepcopy(self.instance.get(name, idx))  # type: ignore

    def get_all(self, name: str) -> Dict[str, Any]:
        """
        get all metadata associated with an ontology
        :param name: name of ontology
        :return:
        """
        return self.instance.database[name]  # type: ignore

    def add(self, name: str, metadata: Dict[str, Any]):
        """
        add metadata to the ontology. Note, metadata is assumed to be static, and global. Calling this function will
        override any existing entries with associated with the keys in the metadata dict
        :param name: name of ontology to add to
        :param metadata: dict in format {idx:metadata}
        :return:
        """
        self.instance.add(name, metadata)  # type: ignore


class SynonymDatabase:
    """
    Singleton of a database of synonyms.
    """

    instance: Optional["__SynonymDatabase"] = None

    class __SynonymDatabase:
        database: DefaultDict[str, Dict[str, List[SynonymData]]] = defaultdict(dict)

        def add(self, name: str, synonyms: Dict[str, List[SynonymData]]):
            for syn_string, syn_data in synonyms.items():
                existing_data = self.database[name].get(syn_string, [])
                for x in filter(lambda x: x not in existing_data, syn_data):
                    existing_data.append(x)
                self.database[name][syn_string] = existing_data

        def get(self, name: str, synonym: str) -> List[SynonymData]:
            return self.database[name].get(synonym, [])

    def __init__(self):
        if not SynonymDatabase.instance:
            SynonymDatabase.instance = SynonymDatabase.__SynonymDatabase()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def get(self, name: str, synonym: str) -> List[SynonymData]:
        """
        get a list of SynonymData associated with an ontology and synonym string
        :param name: name of ontology to query
        :param synonym: idx to query
        :return:
        """
        return copy.copy(self.instance.get(name, synonym))  # type: ignore

    def get_all(self, name: str) -> Dict[str, List[SynonymData]]:
        """
        get all synonyms associated with an ontology
        :param name: name of ontology
        :return:
        """
        return self.instance.database[name]  # type: ignore

    def add(self, name: str, synonyms: Dict[str, List[SynonymData]]):
        """
        add SynonymData to the database.
        :param name: name of ontology to add to
        :param synonyms: dict in format {synonym string:List[SynonymData]}
        :return:
        """
        self.instance.add(name, synonyms)  # type: ignore


class OntologyParser(ABC):
    """
    Parse an ontology (or similar) into a set of outputs suitable for NLP entity linking. This involves generating
    two dataframes: one that holds the linking metadata (e.g. default label, IDX and other info), and another that
    holds any synonym information
    Implementations should have a class attribute 'name' to something suitably representative
    """

    name = "unnamed"
    training_col_names = ["id", "syn1", "syn2"]
    # the synonym table should have these (and only these columns)
    all_synonym_column_names = [IDX, SYN, MAPPING_TYPE]
    # the metadata table should have at least these columns
    minimum_metadata_column_names = [IDX, DEFAULT_LABEL]

    def __init__(self, in_path: str, synonym_generators: Optional[List[SynonymGenerator]] = None):
        """

        :param in_path: Path to some resource that should be processed (e.g. owl file, db config, tsv etc)
        :param synonym_generators: list of synonym generators to apply to this parser
        """
        self.synonym_generator_permutations = (
            self.get_synonym_generator_permutations(synonym_generators)
            if synonym_generators
            else None
        )
        self.in_path = in_path

    def dataframe_to_syndata_dict(self, df: pd.DataFrame) -> Dict[str, List[SynonymData]]:
        df_as_dict = df.groupby(SYN).agg(list).to_dict(orient="index")
        result = defaultdict(list)
        for synonym, metadata_dict in df_as_dict.items():
            for idx, mapping_type_lst in zip(metadata_dict[IDX], metadata_dict[MAPPING_TYPE]):
                result[synonym].append(SynonymData(idx=idx, mapping_type=mapping_type_lst))
        return result

    def populate_metadata_database(self):
        """
        populate the metadata database with this ontology
        :return:
        """
        _, metadata_df = self.generate_synonym_and_metadata_dataframes()
        metadata = metadata_df.to_dict(orient="index")
        MetadataDatabase().add(self.name, metadata)

    def populate_synonym_database(self):
        """
        call synonym generators and populate the synonym database
        :return:
        """
        synonym_df, _ = self.generate_synonym_and_metadata_dataframes()
        synonym_data = self.dataframe_to_syndata_dict(synonym_df)
        generated_synonym_data = self.run_synonym_generators(synonym_data)
        logger.info(
            f"{len(synonym_data)} original synonyms and {len(generated_synonym_data)} generated synonyms produced"
        )

        SynonymDatabase().add(self.name, synonym_data)
        SynonymDatabase().add(self.name, generated_synonym_data)

    def run_synonym_generators(
        self, synonym_data: Dict[str, List[SynonymData]]
    ) -> Dict[str, List[SynonymData]]:
        """
        for every perumation of modifiers, generate a list of syns, then aggregate at the end
        :param synonym_data:
        :return:
        """
        final = {}
        if self.synonym_generator_permutations:
            for i, permutation_list in enumerate(self.synonym_generator_permutations):
                logger.info(
                    f"running permutation set {i} on {self.name}. Permutations: {permutation_list}"
                )
                generated_synonym_data = copy.deepcopy(synonym_data)
                for generator in permutation_list:
                    generated_synonym_data = generator(generated_synonym_data)
                final.update(generated_synonym_data)
        return final

    @cachetools.cached(cache={})
    def generate_synonym_and_metadata_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        splits a table of ontology information into a synonym table and a metadata table, deduplicating and grouping
        as appropriate
        :return: a 2-tuple - first is synonym dataframe, second is metadata
        """
        df = self.parse_to_dataframe()
        # in case the default label isn't populated, just use the IDX
        df.loc[pd.isnull(df[DEFAULT_LABEL]), DEFAULT_LABEL] = df[IDX]
        # ensure correct order
        syn_df = df[self.all_synonym_column_names]
        syn_df.drop_duplicates(subset=self.all_synonym_column_names)
        # group mapping types of same synonym together
        syn_df = syn_df.groupby(by=[IDX, SYN]).agg(set).reset_index()
        syn_df = syn_df.dropna(axis=0)
        syn_df.sort_values(by=[IDX, SYN], inplace=True)
        # needs to be a list so can be serialised
        syn_df[MAPPING_TYPE] = syn_df[MAPPING_TYPE].apply(list)
        syn_df.reset_index(inplace=True, drop=True)
        metadata_columns = df.columns.tolist()
        metadata_columns.remove(MAPPING_TYPE)
        metadata_columns.remove(SYN)
        metadata_df = df[metadata_columns]
        metadata_df = metadata_df.drop_duplicates(subset=[IDX]).dropna(axis=0)
        metadata_df.set_index(inplace=True, drop=True, keys=IDX)
        assert set(OntologyParser.all_synonym_column_names).issubset(syn_df.columns)
        return syn_df, metadata_df

    def parse_to_dataframe(self) -> pd.DataFrame:
        """
        implementations should override this method, returning a 'long, thin' pd.DataFrame of at least the following
        columns:


        [IDX, DEFAULT_LABEL, SYN, MAPPING_TYPE]

        IDX: the ontology id
        DEFAULT_LABEL: the preferred label
        SYN: a synonym of the concept
        MAPPING_TYPE: the type of mapping from default label to synonym - e.g. xref, exactSyn etc. Usually defined by
                    the ontology
        :return:
        """
        raise NotImplementedError()

    def format_training_table(self) -> pd.DataFrame:
        """
        generate a table of synonym pairs. Useful for aligning an embedding space (e.g. as for sapbert)
        :return:
        """
        synonym_table = self.generate_synonym_and_metadata_dataframes()[0]
        tqdm.pandas(desc=f"generating training pairs for {self.name}")
        df = synonym_table.groupby(by=[IDX]).progress_apply(self.select_pos_pairs)
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

    def get_synonym_generator_permutations(
        self, synonym_generators: List[SynonymGenerator]
    ) -> List[Iterable[SynonymGenerator]]:
        result: List[Iterable[SynonymGenerator]] = []
        for i in range(len(synonym_generators)):
            result.extend(itertools.permutations(synonym_generators, i + 1))

        return result


class JsonLinesOntologyParser(OntologyParser):
    """
    A parser for a jsonlines dataset. Assumes one kb entry per line (i.e. json object)
    implemetations should implement json_dict_to_parser_dict (see method notes for details
    """

    def read(self, path: str) -> Iterable[Dict[str, Any]]:
        for json_path in Path(path).glob("*.json"):
            with json_path.open(mode="r") as f:
                for line in f:
                    yield json.loads(line)

    def parse_to_dataframe(self):
        return pd.concat(self.json_dict_to_parser_dataframe(self.read(self.in_path)))

    def json_dict_to_parser_dataframe(
        self, jsons_gen: Iterable[Dict[str, Any]]
    ) -> Iterable[pd.DataFrame]:
        """
        for a given input json (represented as a python dict), yield a pd.DataFrame compatible with the expected
        strucutre of the Ontology Parser superclass - i.e. should have keys for SYN, MAPPING_TYPE, DEFAULT_LABEL and
        IDX. All other keys are used as mapping metadata
        :param jsons_gen: iterator of python dict representing json objects
        :return: pd.DataFrame
        """
        raise NotImplementedError()


class OpenTargetsDiseaseOntologyParser(JsonLinesOntologyParser):
    name = "OPENTARGETS_DISEASE"

    def json_dict_to_parser_dataframe(
        self, jsons_gen: Iterable[Dict[str, Any]]
    ) -> Iterable[pd.DataFrame]:
        # we ignore related syns for now until we decide how the system should handle them
        for json_dict in jsons_gen:
            synonyms = json_dict.get("synonyms", {})
            exact_syns = synonyms.get("hasExactSynonym", [])
            exact_syns.append(json_dict["name"])
            df = pd.DataFrame(exact_syns, columns=[SYN])
            df[MAPPING_TYPE] = "hasExactSynonym"
            df[DEFAULT_LABEL] = json_dict["name"]
            df[IDX] = json_dict.get("code")
            df["dbXRefs"] = [json_dict.get("dbXRefs", [])] * df.shape[0]
            yield df


class OpenTargetsTargetOntologyParser(JsonLinesOntologyParser):
    name = "OPENTARGETS_TARGET"

    def json_dict_to_parser_dataframe(
        self, jsons_gen: Iterable[Dict[str, Any]]
    ) -> Iterable[pd.DataFrame]:
        for json_dict in jsons_gen:
            records = []
            for key in ["synonyms", "obsoleteSymbols", "obsoleteNames", "proteinIds"]:
                synonyms_and_sources_lst = json_dict.get(key, [])
                for record in synonyms_and_sources_lst:
                    if "label" in record and "id" in record:
                        raise RuntimeError(f"record: {record} has both id and label specified")
                    elif "label" in record:
                        record[SYN] = record.pop("label")
                    elif "id" in record:
                        record[SYN] = record.pop("id")
                    record[MAPPING_TYPE] = record.pop("source")
                    records.append(record)

            records.append({SYN: json_dict["approvedSymbol"], MAPPING_TYPE: "approvedSymbol"})
            records.append({SYN: json_dict["id"], MAPPING_TYPE: "opentargets_id"})
            df = pd.DataFrame.from_records(records, columns=[SYN, MAPPING_TYPE])
            df[IDX] = json_dict["id"]
            df[DEFAULT_LABEL] = json_dict["approvedSymbol"]
            df["dbXRefs"] = [json_dict.get("dbXRefs", [])] * df.shape[0]
            yield df


class OpenTargetsMoleculeOntologyParser(JsonLinesOntologyParser):
    name = "OPENTARGETS_MOLECULE"

    def json_dict_to_parser_dataframe(
        self, jsons_gen: Iterable[Dict[str, Any]]
    ) -> Iterable[pd.DataFrame]:
        for json_dict in jsons_gen:
            synonyms = json_dict.get("synonyms", [])
            mapping_types = ["synonyms"] * len(synonyms)
            trade_names = json_dict.get("tradeNames", [])
            synonyms.extend(trade_names)
            mapping_types.extend(["tradeNames"] * len(trade_names))
            cross_references = [json_dict.get("crossReferences", {})] * len(synonyms)
            df = pd.DataFrame(
                {
                    SYN: synonyms,
                    MAPPING_TYPE: mapping_types,
                    "crossReferences": cross_references,
                    DEFAULT_LABEL: json_dict["name"],
                    IDX: json_dict["id"],
                }
            )
            yield df


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

    def parse_to_dataframe(self) -> pd.DataFrame:
        g = rdflib.Graph()
        g.parse(self.in_path)
        label_pred_str = "http://www.w3.org/2000/01/rdf-schema#label"
        label_predicates = URIRef(label_pred_str)
        synonym_predicates = [URIRef(x) for x in self._get_synonym_predicates()]
        default_labels = []
        iris = []
        syns = []
        mapping_type = []

        for sub, obj in g.subject_objects(label_predicates):
            default_labels.append(str(obj))
            iris.append(str(sub))
            syns.append(str(obj))
            mapping_type.append(label_pred_str)
            for syn_predicate in synonym_predicates:
                for other_syn_obj in g.objects(subject=sub, predicate=syn_predicate):
                    default_labels.append(str(obj))
                    iris.append(str(sub))
                    syns.append(str(other_syn_obj))
                    mapping_type.append(syn_predicate)
        df = pd.DataFrame.from_dict(
            {DEFAULT_LABEL: default_labels, IDX: iris, SYN: syns, MAPPING_TYPE: mapping_type}
        )
        return df


class UberonOntologyParser(RDFGraphParser):
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


class MondoOntologyParser(OntologyParser):
    name = "MONDO"
    """
    input should be an MONDO owl file
    e.g.
    https://www.ebi.ac.uk/ols/ontologies/mondo
    """

    def parse_to_dataframe(self) -> pd.DataFrame:
        x = json.load(open(self.in_path, "r"))
        graph = x["graphs"][0]
        nodes = graph["nodes"]
        ids = []
        default_label_list = []
        all_syns = []
        mapping_type = []
        for i, node in enumerate(nodes):
            idx = node["id"]
            default_label = node.get("lbl")
            # add default_label to syn type
            all_syns.append(default_label)
            default_label_list.append(default_label)
            mapping_type.append("lbl")
            ids.append(idx)

            syns = node.get("meta", {}).get("synonyms", [])
            for syn_dict in syns:
                pred = syn_dict["pred"]
                mapping_type.append(pred)
                syn = syn_dict["val"]
                ids.append(idx)
                default_label_list.append(default_label)
                all_syns.append(syn)

        df = pd.DataFrame.from_dict(
            {IDX: ids, DEFAULT_LABEL: default_label_list, SYN: all_syns, MAPPING_TYPE: mapping_type}
        )
        return df


class EnsemblOntologyParser(OntologyParser):
    name = "ENSEMBL"
    """
    input is a json from HGNC
     e.g. http://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/json/hgnc_complete_set.json
    :return:
    """

    def parse_to_dataframe(self) -> pd.DataFrame:

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
        all_mapping_type: List[str] = []
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
                synonyms: List[Tuple[str, str]] = []
                for hgnc_key in keys_to_check:
                    synonyms_this_entity = get_with_default_list(hgnc_key)
                    for potential_synonym in synonyms_this_entity:
                        synonyms.append(
                            (potential_synonym, hgnc_key),
                        )

                synonyms = list(set(synonyms))
                # filter any very short matches
                synonyms_and_mapping_type = [x for x in synonyms if len(x[0]) > 2]
                synonyms_strings = []
                for synonym_str, mapping_t in synonyms_and_mapping_type:
                    all_mapping_type.append(mapping_t)
                    synonyms_strings.append(synonym_str)

                num_syns = len(synonyms_strings)
                ids.extend([ensembl_gene_id] * num_syns)
                default_label.extend([name] * num_syns)
                all_syns.extend(synonyms_strings)

        df = pd.DataFrame.from_dict(
            {IDX: ids, DEFAULT_LABEL: default_label, SYN: all_syns, MAPPING_TYPE: all_mapping_type}
        )
        return df


class ChemblOntologyParser(OntologyParser):
    name = "CHEMBL"
    """
    input is a sqllite dump from Chembl, e.g.
    https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_29_sqlite.tar.gz
    :return:
    """

    def parse_to_dataframe(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.in_path)
        query = f"""
            SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, synonyms AS {SYN}, syn_type AS {MAPPING_TYPE} 
            FROM molecule_dictionary AS md
                     JOIN molecule_synonyms ms ON md.molregno = ms.molregno
            UNION ALL
            SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, pref_name AS {SYN}, "pref_name" AS {MAPPING_TYPE} 
            FROM molecule_dictionary
        """  # noqa
        df = pd.read_sql(query, conn)
        # eliminate anything without a pref_name, as will be too big otherwise
        df = df.dropna(subset=[DEFAULT_LABEL])
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

    def parse_to_dataframe(self) -> pd.DataFrame:

        ids = []
        default_labels = []
        all_syns = []
        mapping_type = []
        with open(self.in_path, "r") as f:
            id = ""
            default_label = ""
            for line in f:
                text = line.rstrip()
                if text.startswith("id:"):
                    id = text.split(" ")[1]
                elif text.startswith("name:"):
                    default_label = text.split(" ")[1][1:]
                    ids.append(id)
                    default_labels.append(default_label)
                    all_syns.append(default_label)
                    mapping_type.append("name")
                elif text.startswith("synonym:"):
                    syn = text.split(" ")[1][1:-1]
                    mapping = text.split(" ")[2]
                    ids.append(id)
                    default_labels.append(default_label)
                    all_syns.append(syn)
                    mapping_type.append(mapping)
                else:
                    pass
        df = pd.DataFrame.from_dict(
            {IDX: ids, DEFAULT_LABEL: default_labels, SYN: all_syns, MAPPING_TYPE: mapping_type}
        )
        return df


class MeddraOntologyParser(OntologyParser):
    name = "MEDDRA"
    """
    input is an unzipped directory to a MEddra release (Note, requires licence). This
    should contain the files 'mdhier.asc' and 'llt.asc'
    :return:
    """

    def parse_to_dataframe(self) -> pd.DataFrame:
        # hierarchy path
        mdheir_path = os.path.join(self.in_path, "mdhier.asc")
        # low level term path
        llt_path = os.path.join(self.in_path, "llt.asc")
        hier_df = pd.read_csv(mdheir_path, sep="$", header=None)
        hier_df.columns = [
            "pt_code",
            "hlt_code",
            "hlgt_code",
            "soc_code",
            "pt_name",
            "hlt_name",
            "hlgt_name",
            "soc_name",
            "soc_abbrev",
            "null_field",
            "pt_soc_code",
            "primary_soc_fg",
            "NULL",
        ]

        llt_df = pd.read_csv(llt_path, sep="$", header=None)
        llt_df = llt_df.T.dropna().T
        llt_df.columns = ["llt_code", "llt_name", "pt_code", "llt_currency"]

        ids = []
        default_labels = []
        all_syns = []
        mapping_type = []

        for i, row in hier_df.iterrows():
            idx = row["pt_code"]
            pt_name = row["pt_name"]
            llts = llt_df[llt_df["pt_code"] == idx]
            ids.append(idx)
            default_labels.append(pt_name)
            all_syns.append(pt_name)
            mapping_type.append("meddra_link")
            for j, llt_row in llts.iterrows():
                ids.append(idx)
                default_labels.append(pt_name)
                all_syns.append(llt_row["llt_name"])
                mapping_type.append("meddra_link")
        df = pd.DataFrame.from_dict(
            {IDX: ids, DEFAULT_LABEL: default_labels, SYN: all_syns, MAPPING_TYPE: mapping_type}
        )
        return df
