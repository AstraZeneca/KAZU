import copy
import itertools
import json
import logging
import os
import re
import sqlite3
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterable, Set, FrozenSet
from typing import Optional, DefaultDict
from urllib import parse
import cachetools
import pandas as pd
import pydash
import rdflib
from rdflib import URIRef
from tqdm.auto import tqdm

# dataframe column keys
from kazu.modelling.ontology_preprocessing.synonym_generation import (
    CombinatorialSynonymGenerator,
    GreekSymbolSubstitution,
)
from kazu.data.data import SynonymData

DEFAULT_LABEL = "default_label"
IDX = "idx"
SYN = "syn"
MAPPING_TYPE = "mapping_type"
SOURCE = "source"

logger = logging.getLogger(__name__)


class StringNormalizer:
    """
    normalise a biomedical string for search
    TODO: make configurable
    """

    allowed_additional_chars = {" ", "(", ")", "+", "-", "‐"}
    greek_subs = GreekSymbolSubstitution.GREEK_SUBS
    greek_subs_upper = {x: f" {y.upper()} " for x, y in greek_subs.items()}
    other_subs = {
        "(": " (",
        ")": ") ",
        ",": " ",
        "/": " ",
        "VIII": " 8 ",
        "VII": " 7 ",
        "XII": " 12 ",
        "III": " 3 ",
        "VI": " 6 ",
        "IV": " 4 ",
        "IX": " 9 ",
        "XI": " 11 ",
        "II": " 2 ",
    }
    re_subs = {
        re.compile(r"(?<!\()-(?!\))"): " ",  # minus not in brackets
        re.compile(r"(?<!\()‐(?!\))"): " ",  # hyphen not in brackets
        re.compile(r"\sI\s|\sI$"): " 1 ",
        re.compile(r"\sV\s|\sV$"): " 5 ",
        re.compile(r"\sX\s|\sX$"): " 10 ",
    }
    re_subs_2 = {
        re.compile(r"\sA\s|\sA$|^A\s"): " ALPHA ",
        re.compile(r"\sB\s|\sB$|^B\s"): " BETA ",
    }

    number_split_pattern = re.compile(r"(\d+)")

    symbol_number_split = re.compile(r"(\d+)$")
    trailing_lowercase_s_split = re.compile(r"(.*)(s)$")

    @staticmethod
    def unigram_bigram_tokenize(norm_string: str):
        return StringNormalizer.tokenize(norm_string, {1, 2})

    @staticmethod
    def tokenize(norm_string: str, n_set: Set[int]) -> List[str]:
        parts = norm_string.split(" ")
        result = []
        for n in n_set:
            result.extend(zip(*[parts[i:] for i in range(n)]))
        return [" ".join(ngram) for ngram in result]

    @staticmethod
    def is_symbol_like(debug, original_string) -> Optional[str]:
        # True if all upper, all alphanum, no spaces,

        for char in original_string:
            if char.islower() or not char.isalnum():
                return None
        else:
            splits = [
                x.strip() for x in re.split(StringNormalizer.symbol_number_split, original_string)
            ]
            string = " ".join(splits).strip()
            if debug:
                print(string)
            return string

    @staticmethod
    def split_on_trailing_s_prefix(debug, string):
        splits = [x.strip() for x in re.split(StringNormalizer.trailing_lowercase_s_split, string)]
        string = " ".join(splits).strip()
        if debug:
            print(string)
        return string

    @staticmethod
    def normalize(original_string: str, debug: bool = False):
        original_string = original_string.strip()
        symbol_like = StringNormalizer.is_symbol_like(debug, original_string)
        if symbol_like:
            return symbol_like
        else:
            string = StringNormalizer.replace_substrings(debug, original_string)

            # split up numbers
            string = StringNormalizer.split_on_numbers(debug, string)
            # replace greek
            string = StringNormalizer.replace_greek(debug, string)

            # strip non alphanum
            string = StringNormalizer.replace_non_alphanum(debug, string)

            string = StringNormalizer.split_on_trailing_s_prefix(debug, string)
            # strip modifying lowercase prefixes
            string = StringNormalizer.handle_lower_case_prefixes(debug, string)

            string = StringNormalizer.sub_greek_char_abbreviations(debug, string)

            string = string.strip()
            if debug:
                print(string)
            return string

    @staticmethod
    def sub_greek_char_abbreviations(debug, string):
        for re_sub, replace in StringNormalizer.re_subs_2.items():
            string = re.sub(re_sub, replace, string)
            if debug:
                print(string)
        return string

    @staticmethod
    def to_upper(debug, string):
        string = string.upper()
        if debug:
            print(string)
        return string

    @staticmethod
    def handle_lower_case_prefixes(debug, string):
        """
        preserve case only if first char of contiguous subsequence is lower case, and is alphanum, and upper
        case detected in rest of part
        :param debug:
        :param string:
        :return:
        """
        parts = string.split(" ")
        new_parts = []
        for part in parts:
            if part != "":
                if part.islower() and not len(part) == 1:
                    new_parts.append(part.upper())
                else:
                    first_char_case = part[0].islower()
                    if (first_char_case and part[0].isalnum()) or (
                        first_char_case and len(part) == 1
                    ):
                        new_parts.append(part)
                    else:
                        new_parts.append(part.upper())
        string = " ".join(new_parts)
        if debug:
            print(string)
        return string

    @staticmethod
    def replace_non_alphanum(debug, string):
        string = "".join(
            [x for x in string if (x.isalnum() or x in StringNormalizer.allowed_additional_chars)]
        )
        if debug:
            print(string)
        return string

    @staticmethod
    def replace_greek(debug, string):
        for substr, replace in StringNormalizer.greek_subs_upper.items():
            if substr in string:
                string = string.replace(substr, replace)
                if debug:
                    print(string)
        return string

    @staticmethod
    def split_on_numbers(debug, string):
        splits = [x.strip() for x in re.split(StringNormalizer.number_split_pattern, string)]
        string = " ".join(splits)
        if debug:
            print(string)
        return string

    @staticmethod
    def replace_substrings(debug, original_string):
        string = original_string
        # replace substrings
        for substr, replace in StringNormalizer.other_subs.items():
            if substr in string:
                string = string.replace(substr, replace)
                if debug:
                    print(string)
        for re_sub, replace in StringNormalizer.re_subs.items():
            string = re.sub(re_sub, replace, string)
            if debug:
                print(string)
        return string


class MetadataDatabase:
    """
    Singleton of Ontology metadata database. Purpose: metadata needs to be looked up in different linking processes,
    and this singleton allows us to load it once/reduce memory usage
    """

    instance: Optional["__MetadataDatabase"] = None

    class __MetadataDatabase:
        database: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)
        keys_lst: DefaultDict[str, List[str]] = defaultdict(list)

        def add(self, name: str, metadata: Dict[str, Dict[str, Any]]):
            self.database[name].update(metadata)
            self.keys_lst[name] = list(self.database[name].keys())

        def get_by_idx(self, name: str, idx: str) -> Dict[str, Any]:
            return self.database[name][idx]

        def get_by_index(self, name: str, i: int) -> Tuple[str, Dict[str, Any]]:
            idx = self.keys_lst[name][i]
            return idx, self.database[name][idx]

    def __init__(self):
        if not MetadataDatabase.instance:
            MetadataDatabase.instance = MetadataDatabase.__MetadataDatabase()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def get_by_idx(self, name: str, idx: str) -> Dict[str, Any]:
        """
        get the metadata associated with an ontology and id
        :param name: name of ontology to query
        :param idx: idx to query
        :return:
        """
        return copy.deepcopy(self.instance.get_by_idx(name, idx))  # type: ignore

    def get_by_index(self, name: str, i: int) -> Dict:

        return copy.deepcopy(self.instance.get_by_index(name, i))  # type: ignore

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
        syns_database_by_syn: DefaultDict[str, Dict[str, Set[SynonymData]]] = defaultdict(
            lambda: defaultdict(set)
        )
        syns_database_by_idx: DefaultDict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        syns_database_by_syn_global: DefaultDict[str, Set[SynonymData]] = defaultdict(set)
        kb_database_by_syn_global: DefaultDict[str, Set[str]] = defaultdict(set)
        loaded_kbs: Set[str] = set()

        def add(self, name: str, synonyms: Dict[str, FrozenSet[SynonymData]], norm: bool):
            self.loaded_kbs.add(name)
            for syn_string, syn_data_set in synonyms.items():
                if norm:
                    syn_string_norm = StringNormalizer.normalize(syn_string)
                else:
                    syn_string_norm = syn_string
                self.syns_database_by_syn[name][syn_string_norm].update(syn_data_set)
                self.syns_database_by_syn_global[syn_string_norm].update(syn_data_set)
                self.kb_database_by_syn_global[syn_string_norm].add(name)

                for syn_data in syn_data_set:
                    for idx in syn_data.ids:
                        self.syns_database_by_idx[name][idx].add(syn_string_norm)

        def get(self, name: str, synonym: str) -> Set[SynonymData]:
            return self.syns_database_by_syn[name][synonym]

        def get_syns_for_id(self, name: str, idx: str) -> Set[str]:
            return self.syns_database_by_idx[name][idx]

    def __init__(self):
        if not SynonymDatabase.instance:
            SynonymDatabase.instance = SynonymDatabase.__SynonymDatabase()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def get(self, name: str, synonym: str) -> Set[SynonymData]:
        """
        get a list of SynonymData associated with an ontology and synonym string
        :param name: name of ontology to query
        :param synonym: idx to query
        :return:
        """
        return self.instance.get(name, synonym)  # type: ignore

    def get_syns_for_id(self, name: str, idx: str) -> Set[str]:
        return self.instance.get_syns_for_id(name, idx)  # type: ignore

    def get_syns_for_synonym(self, name: str, synonym: str) -> Set[str]:
        """
        get all other syns for a synonym in a kb
        :param name:
        :param idx:
        :return:
        """
        result = set()
        for syn_data in self.get(name, synonym):
            for idx in syn_data.ids:
                result.update(self.get_syns_for_id(name, idx))
        return result

    def get_all(self, name: str) -> Dict[str, Set[SynonymData]]:
        """
        get all synonyms associated with an ontology
        :param name: name of ontology
        :return:
        """
        return self.instance.syns_database_by_syn[name]  # type: ignore

    def get_syns_global(self, synonym: str) -> Set[SynonymData]:
        """
        return a global view of synonym data across all dbs, for a specific synonym
        :return:
        """
        return self.instance.syns_database_by_syn_global.get(synonym, set())

    def get_kbs_for_syn_global(self, synonym: str) -> Set[str]:
        """
        return a global list ok kbs across all dbs, for a specific synonym
        :return:
        """
        return self.instance.kb_database_by_syn_global.get(synonym, set())

    def get_loaded_kbs(self) -> Set[str]:
        """
        return a global view of all ambiguous synonyms data across all dbs
        :return:
        """
        return self.instance.loaded_kbs

    def get_database(self) -> DefaultDict[str, Dict[str, Set[SynonymData]]]:
        return self.instance.syns_database_by_syn  # type: ignore

    def add(self, name: str, synonyms: Dict[str, Set[SynonymData]]):
        """
        add SynonymData to the database.
        :param name: name of ontology to add to
        :param synonyms: dict in format {synonym string:List[SynonymData]}
        :return:
        """
        self.instance.add(name, synonyms, norm=False)  # type: ignore

    def normalise_and_add(self, name: str, synonyms: Dict[str, Set[SynonymData]]):
        """
        add SynonymData to the database.
        :param name: name of ontology to add to
        :param synonyms: dict in format {synonym string:List[SynonymData]}
        :return:
        """
        self.instance.add(name, synonyms, norm=True)  # type: ignore


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

    def __init__(
        self,
        in_path: str,
        is_composite: bool = False,
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
    ):
        """
        :param in_path: Path to some resource that should be processed (e.g. owl file, db config, tsv etc)
        :param is_composite: does the source contain mappings to other ontologies? if so, this affects how we aggregate synonyms
        :param synonym_generators: list of synonym generators to apply to this parser
        """
        self.is_composite = is_composite
        self.synonym_generator = synonym_generator
        self.in_path = in_path

    def find_kb(self, string: str) -> str:
        """
        split an IDX somehow to find the KB reference. only required if self.is_composite
        :param df:
        :return:
        """
        raise NotImplementedError()

    def dataframe_to_syndata_dict(
        self, synonym_df: pd.DataFrame, normalise_original_syns: bool
    ) -> Dict[str, Set[SynonymData]]:
        if self.is_composite:
            result = self.resolve_composite_synonym_dataframe(synonym_df, normalise_original_syns)
        else:
            result = self.resolve_singular_synonym_dataframe(synonym_df, normalise_original_syns)

        return dict(result)

    def resolve_singular_synonym_dataframe(self, df, normalise_original_syns: bool):
        all_syn_data = {}
        for idx, mapping_type_dict in (
            df[[IDX, MAPPING_TYPE]].groupby(IDX).agg(list).to_dict(orient="index").items()
        ):
            unique_mappings = frozenset(pydash.flatten(mapping_type_dict[MAPPING_TYPE]))
            all_syn_data[idx] = SynonymData(ids=frozenset([idx]), mapping_type=unique_mappings)
        result = defaultdict(set)
        for i, row in df[[SYN, IDX]].drop_duplicates().iterrows():
            idx = row[IDX]
            syn = StringNormalizer.normalize(row[SYN]) if normalise_original_syns else row[SYN]
            result[syn].add(all_syn_data[idx])
        return result

    def resolve_composite_synonym_dataframe(
        self, synonym_df: pd.DataFrame, normalise_original_syns: bool
    ):
        synonym_df["composed_ontologies"] = synonym_df[IDX].apply(self.find_kb)
        result = defaultdict(set)
        for i, row in (
            synonym_df[["composed_ontologies", SYN, IDX]]
            .groupby([SYN])
            .agg(set)
            .reset_index()
            .iterrows()
        ):

            syn = StringNormalizer.normalize(row[SYN]) if normalise_original_syns else row[SYN]
            ontologies = row["composed_ontologies"]
            ids = row[IDX]
            if len(ontologies) == 1:
                # most common - one or more ids and one kb per syn
                for idx in ids:
                    result[syn].add(SynonymData(ids=frozenset([idx]), mapping_type=frozenset()))
            elif len(ontologies) == len(ids):
                # pathological scenario - multiple kb ids, one syn. Syn may refer to different concepts (although probably not)
                result[syn].add(SynonymData(ids=frozenset(ids), mapping_type=frozenset()))
                logger.warning(
                    f"found independent identifiers for {syn}: {ids}. This may cause disambiguation problems "
                    f"if the identifiers are not cross references"
                )
            else:
                # pathological scenario - one synonym maps to multiple KBs and multiple ids within those KBs,
                # within the same composite KB
                if len(syn) > 4:
                    # if it's more than 4 chars we'll take a punt that it's probably fine to agg in one id
                    # TODO: how about not taking a punt?
                    result[syn].add(SynonymData(ids=frozenset(ids), mapping_type=frozenset()))
                    logger.warning(
                        f"could not resolve {syn} for {self.name}. ids: {ids}, ontologies: {ontologies}. "
                        f"As {syn} is more than 4 chars, parser has aggregated - i.e. assumed they refer to the same concepts"
                    )
                else:
                    for idx in ids:
                        result[syn].add(SynonymData(ids=frozenset([idx]), mapping_type=frozenset()))
                    logger.warning(
                        f"could not resolve {syn} for {self.name}. ids: {ids}, ontologies: {ontologies}. "
                        f"As {syn} is les than 4 chars, parser has split - i.e. assumed they refer to different concepts"
                    )
        return result

    def populate_metadata_database(self):
        """
        populate the metadata database with this ontology
        :return:
        """
        _, metadata_df = self.generate_synonym_and_metadata_dataframes()
        metadata = metadata_df.to_dict(orient="index")
        MetadataDatabase().add(self.name, metadata)

    def collect_aggregate_synonym_data(
        self, normalise_original_syns: bool
    ) -> Dict[str, Set[SynonymData]]:
        synonym_df, _ = self.generate_synonym_and_metadata_dataframes()
        # strip trailing whitespace from syns
        synonym_df[SYN] = synonym_df[SYN].apply(str.strip)
        synonym_data = self.dataframe_to_syndata_dict(synonym_df, normalise_original_syns)
        return synonym_data

    def generate_synonyms(self):
        """

        :param normalise_original_syns: should the string normaliser be used before aggregating synonym data
        :return:
        """
        synonym_data = self.collect_aggregate_synonym_data(False)
        generated_synonym_data = {}
        if self.synonym_generator:
            generated_synonym_data = self.synonym_generator(synonym_data)
        logger.info(
            f"{len(synonym_data)} original synonyms and {len(generated_synonym_data)} generated synonyms produced"
        )
        return generated_synonym_data

    def populate_synonym_database(self):
        """
        deprecated
        call synonym generators and populate the synonym database
        :return:
        """
        synonym_data = self.collect_aggregate_synonym_data(True)
        # SynonymDatabase().add(self.name, generated_synonym_data)
        SynonymDatabase().add(self.name, synonym_data)

    @cachetools.cached(cache={})
    def generate_synonym_and_metadata_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        splits a table of ontology information into a synonym table and a metadata table, deduplicating and grouping
        as appropriate
        :return: a 2-tuple - first is synonym dataframe, second is metadata
        """
        df = self.parse_to_dataframe()
        df[IDX] = df[IDX].astype(str)

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

    def find_kb(self, string: str) -> str:
        return string.split("_")[0]

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
            df[IDX] = self.look_for_mondo(json_dict["id"], json_dict.get("dbXRefs", []))
            df["dbXRefs"] = [json_dict.get("dbXRefs", [])] * df.shape[0]
            yield df

    def look_for_mondo(self, ot_id: str, db_xrefs: List[str]):
        if "MONDO" in ot_id:
            return ot_id
        for x in db_xrefs:
            if "MONDO" in x:
                return x.replace(":", "_")
        return ot_id


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
            records.append({SYN: json_dict["approvedName"], MAPPING_TYPE: "approvedName"})
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

    @property
    @classmethod
    @abstractmethod
    def _uri_regex(cls):
        """
        subclasses should provide this as a class attribute.

        It should be a compiled regex object that matches on valid URIs for the ontology
        being parsed."""
        pass

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
            if not self.is_valid_iri(str(sub)):
                continue

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

    def is_valid_iri(self, text: str) -> bool:
        """
        Check if input string is a valid IRI for the ontology being parsed.

        Uses self._uri_regex to define valid IRIs
        """
        match = self._uri_regex.match(text)
        return bool(match)


class UberonOntologyParser(RDFGraphParser):
    name = "UBERON"
    _uri_regex = re.compile("^http://purl.obolibrary.org/obo/UBERON_[0-9]+$")
    """
    input should be an UBERON owl file
    e.g.
    https://www.ebi.ac.uk/ols/ontologies/uberon
    """

    def _get_synonym_predicates(self) -> List[str]:
        return [
            # "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym",
            "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
        ]


class MondoOntologyParser(OntologyParser):
    name = "MONDO"
    _uri_regex = re.compile("^http://purl.obolibrary.org/obo/(MONDO|HP)_[0-9]+$")
    """
    input should be an MONDO json file
    e.g.
    https://www.ebi.ac.uk/ols/ontologies/mondo
    """

    def find_kb(self, string: str):
        return parse.urlparse(string).path.split("_")[0]

    def parse_to_dataframe(self) -> pd.DataFrame:
        x = json.load(open(self.in_path, "r"))
        graph = x["graphs"][0]
        nodes = graph["nodes"]
        ids = []
        default_label_list = []
        all_syns = []
        mapping_type = []
        for i, node in enumerate(nodes):
            if not self.is_valid_iri(node["id"]):
                continue

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
                if pred in {"hasExactSynonym"}:
                    mapping_type.append(pred)
                    syn = syn_dict["val"]
                    ids.append(idx)
                    default_label_list.append(default_label)
                    all_syns.append(syn)

        df = pd.DataFrame.from_dict(
            {IDX: ids, DEFAULT_LABEL: default_label_list, SYN: all_syns, MAPPING_TYPE: mapping_type}
        )
        return df

    def is_valid_iri(self, text: str) -> bool:
        match = self._uri_regex.match(text)
        return bool(match)


class EnsemblOntologyParser(OntologyParser):
    name = "ENSEMBL"
    """
    input is a json from HGNC
     e.g. http://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/json/hgnc_complete_set.json
    :return:
    """

    def __init__(self, in_path: str, additional_syns_path: str):

        super().__init__(in_path)
        with open(additional_syns_path, "r") as f:

            self.additional_syns = json.load(f)

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
                        synonyms.extend(
                            (potential_synonym, hgnc_key)
                            for potential_synonym in synonyms_this_entity
                        )

                synonyms = list(set(synonyms))
                synonyms_strings = []
                for synonym_str, mapping_t in synonyms:
                    all_mapping_type.append(mapping_t)
                    synonyms_strings.append(synonym_str)

                # also include any additional synonyms we've defined
                additional_syns = self.additional_syns["additional_syns"].get(ensembl_gene_id, [])
                for additional_syn in additional_syns:
                    synonyms_strings.append(additional_syn)
                    all_mapping_type.append("kazu_curated")

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
    _uri_regex = re.compile("^http://purl.obolibrary.org/obo/CLO_[0-9]+$")
    """
    input is a CLO Owl file
    https://www.ebi.ac.uk/ols/ontologies/clo
    """

    def _get_synonym_predicates(self) -> List[str]:
        return [
            # "http://purl.obolibrary.org/obo/hasNarrowSynonym",
            "http://purl.obolibrary.org/obo/hasExactSynonym",
            # "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym",
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
                    default_label = text[5:].strip()
                    ids.append(id)
                    default_labels.append(default_label)
                    all_syns.append(default_label)
                    mapping_type.append("name")
                elif text.startswith("synonym:"):
                    match = self._synonym_regex.match(text)
                    if match is None:
                        raise ValueError(
                            """synonym line does not match our synonym regex.
                            Either something is wrong with the file, or it has updated
                            and our regex is not correct/general enough."""
                        )
                    ids.append(id)
                    default_labels.append(default_label)
                    all_syns.append(match.group("syn"))
                    mapping_type.append(match.group("mapping"))
                else:
                    pass
        df = pd.DataFrame.from_dict(
            {IDX: ids, DEFAULT_LABEL: default_labels, SYN: all_syns, MAPPING_TYPE: mapping_type}
        )
        return df

    _synonym_regex = re.compile(
        r"""^synonym:      # line that begins synonyms
        \s*                # any amount of whitespace (standardly a single space)
        "(?P<syn>[^"]*)"   # a quoted string - capture this as a named match group 'syn'
        \s*                # any amount of separating whitespace (standardly a single space)
        (?P<mapping>\w*)   # a sequence of word characters representing the mapping type
        \s*                # any amount of separating whitespace (standardly a single space)
        \[\]               # an open and close bracket at the end of the string
        $""",
        re.VERBOSE,
    )


class MeddraOntologyParser(OntologyParser):
    name = "MEDDRA"
    """
    input is an unzipped directory to a MEddra release (Note, requires licence). This
    should contain the files 'mdhier.asc' and 'llt.asc'
    :return:
    """

    _mdhier_asc_col_names = (
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
    )

    _llt_asc_column_names = (
        "llt_code",
        "llt_name",
        "pt_code",
        "llt_whoart_code",
        "llt_harts_code",
        "llt_costart_sym",
        "llt_icd9_code",
        "llt_icd9cm_code",
        "llt_icd10_code",
        "llt_currency",
        "llt_jart_code",
        "NULL",
    )

    def parse_to_dataframe(self) -> pd.DataFrame:
        # hierarchy path
        mdheir_path = os.path.join(self.in_path, "mdhier.asc")
        # low level term path
        llt_path = os.path.join(self.in_path, "llt.asc")
        hier_df = pd.read_csv(
            mdheir_path,
            sep="$",
            header=None,
            names=self._mdhier_asc_col_names,
            dtype="string",
        )

        llt_df = pd.read_csv(
            llt_path,
            sep="$",
            header=None,
            names=self._llt_asc_column_names,
            usecols=("llt_name", "pt_code"),
            dtype="string",
        )
        llt_df = llt_df.dropna(axis=1)

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

        for i, row in hier_df[["hlt_code", "hlt_name"]].drop_duplicates().iterrows():
            ids.append(row["hlt_code"])
            default_labels.append(row["hlt_name"])
            all_syns.append(row["hlt_name"])
            mapping_type.append("meddra_link")
        for i, row in hier_df[["hlgt_code", "hlgt_name"]].drop_duplicates().iterrows():
            ids.append(row["hlgt_code"])
            default_labels.append(row["hlgt_name"])
            all_syns.append(row["hlgt_name"])
            mapping_type.append("meddra_link")
        df = pd.DataFrame.from_dict(
            {IDX: ids, DEFAULT_LABEL: default_labels, SYN: all_syns, MAPPING_TYPE: mapping_type}
        )
        return df
