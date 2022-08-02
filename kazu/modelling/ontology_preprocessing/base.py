import itertools
import json
import logging
import os
import re
import sqlite3
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterable, Set, Optional, FrozenSet
from urllib import parse

import pandas as pd
import rdflib
from kazu.data.data import (
    EquivalentIdSet,
    EquivalentIdAggregationStrategy,
    SynonymTerm,
    SimpleValue,
)

# dataframe column keys
from kazu.modelling.database.in_memory_db import MetadataDatabase, SynonymDatabase
from kazu.modelling.language.symbol_classification import SymbolClassifier, DefaultSymbolClassifier
from kazu.modelling.ontology_preprocessing.synonym_generation import (
    CombinatorialSynonymGenerator,
)
from kazu.utils.spacy_pipeline import SpacyPipeline
from kazu.utils.string_normalizer import StringNormalizer
from rdflib import URIRef

DEFAULT_LABEL = "default_label"
IDX = "idx"
SYN = "syn"
MAPPING_TYPE = "mapping_type"
SOURCE = "source"
DATA_ORIGIN = "data_origin"


logger = logging.getLogger(__name__)


class OntologyParser(ABC):
    """
    Parse an ontology (or similar) into a set of outputs suitable for NLP entity linking. This involves generating
    two dataframes: one that holds the linking metadata (e.g. default label, IDX and other info), and another that
    holds any synonym information
    Implementations should have a class attribute 'name' to something suitably representative.
    Note: It is the responsibility of a parser implementation to add default labels as synonyms.
    """

    name = "unnamed"  # a label for this parser
    # the synonym table should have these (and only these columns)
    all_synonym_column_names = [IDX, SYN, MAPPING_TYPE]
    # the metadata table should have at least these columns (note, IDX will become the index)
    minimum_metadata_column_names = [DEFAULT_LABEL, DATA_ORIGIN]

    def __init__(
        self,
        in_path: str,
        spacy_pipeline: Optional[SpacyPipeline] = None,
        synonym_merge_threshold: float = 0.65,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        symbol_classifier: Optional[SymbolClassifier] = None,
    ):
        """
        :param in_path: Path to some resource that should be processed (e.g. owl file, db config, tsv etc)
        :param spacy_pipeline: Optional instance of SpacyPipeline.  Used for resolving ambiguous symbolic synonyms via
            similarity calculation of the default label associated with the conflicted labels. If no instance is
            provided, all synonym conflicts will be merged. This is not recommended!
        :param synonym_merge_threshold: similarity threshold to trigger a merge of conflicted synonyms into a single
            EquivalentIdSet
        :param data_origin: The origin of this dataset - e.g. HGNC release 2.1, MEDDRA 24.1 etc. Note, this is different from the
            parser.name, as is used to identify the origin of a mapping back to a data source
        :param synonym_generator: optional CombinatorialSynonymGenerator
        :param symbol_classifier: optional SymbolClassifier. When parsing a data source, synonyms that are
            symbolic (as determined by the classifier) that refer to more than one id are more likely to be ambiguous.
            Therefore, we assume they refer to unique concepts (e.g. COX 1 could be 'ENSG00000095303' OR
            'ENSG00000198804', and thus they will yield multiple instances of EquivalentIdSet.
            Non symbolic synonyms (i.e. noun phrases) are far less likely to refer to distinct entities, so we merge
            all associated ID's into a single EquivalentIdSet. If no symbol classifier is specified, a default one will
            be used
        """
        if spacy_pipeline is None:
            logger.warning("no spacy pipeline configured. Synonym resolution disabled.")
        self.spacy_pipeline = spacy_pipeline
        self.synonym_merge_threshold = synonym_merge_threshold
        self.data_origin = data_origin
        self.synonym_generator = synonym_generator
        self.in_path = in_path
        self.parsed_dataframe: Optional[pd.DataFrame] = None
        self.symbol_classifier: SymbolClassifier = (
            symbol_classifier if symbol_classifier is not None else DefaultSymbolClassifier()
        )
        self.metadata_db = MetadataDatabase()

    def find_kb(self, string: str) -> str:
        """
        split an IDX somehow to find the ontology SOURCE reference
        :param string: the IDX string to process
        :return:
        """
        raise NotImplementedError()

    def resolve_synonyms(self, synonym_df: pd.DataFrame) -> Set[SynonymTerm]:
        """
        synonym lists are noisy, so we need an algorithm to identify when a synonym

        """

        result = set()
        synonym_df["syn_norm"] = synonym_df[SYN].apply(StringNormalizer.normalize)

        for i, row in (
            synonym_df[["syn_norm", SYN, IDX, MAPPING_TYPE]]
            .groupby(["syn_norm"])
            .agg(set)
            .reset_index()
            .iterrows()
        ):

            syn_set = row[SYN]
            mapping_type_set: FrozenSet[str] = frozenset(row[MAPPING_TYPE])
            syn_norm = row["syn_norm"]
            if len(syn_set) > 1:
                logger.debug(f"normaliser has merged {syn_set} into a single term: {syn_norm}")

            is_symbolic = any(self.symbol_classifier.is_symbolic(x) for x in syn_set)

            ids: Set[str] = row[IDX]
            id_to_source = {}
            ontologies = set()
            for idx in ids:
                source = self.find_kb(idx)
                ontologies.add(source)
                id_to_source[idx] = source

            associated_id_sets = self.score_and_group_ids(ids, id_to_source, is_symbolic, syn_set)

            synonym_term = SynonymTerm(
                term_norm=syn_norm,
                terms=frozenset(syn_set),
                is_symbolic=is_symbolic,
                mapping_types=mapping_type_set,
                associated_id_sets=associated_id_sets,
                parser_name=self.name,
            )

            result.add(synonym_term)

        return result

    def score_and_group_ids(
        self,
        ids: Set[str],
        id_to_source: Dict[str, str],
        is_symbolic: bool,
        original_syn_set: Set[str],
    ) -> FrozenSet[EquivalentIdSet]:
        if not isinstance(self.spacy_pipeline, SpacyPipeline):
            associated_id_sets = frozenset(
                (
                    EquivalentIdSet(
                        ids=frozenset(ids),
                        aggregated_by=EquivalentIdAggregationStrategy.NO_STRATEGY,
                        ids_to_source={idx: id_to_source[idx] for idx in ids},
                    ),
                )
            )
            return associated_id_sets
        else:

            id_to_label = {
                idx: str(self.metadata_db.get_by_idx(self.name, idx)[DEFAULT_LABEL]) for idx in ids
            }

            if len(id_to_label) == 1 or not is_symbolic:
                # default label is the same, or it's not symbolic. It's the same concept
                return frozenset(
                    (
                        EquivalentIdSet(
                            ids=frozenset(ids),
                            aggregated_by=EquivalentIdAggregationStrategy.UNAMBIGUOUS,
                            ids_to_source={idx: id_to_source[idx] for idx in ids},
                        ),
                    )
                )
            else:
                # use similarity to group ids into EquivalentIdSets
                Ids = Set[str]
                DefaultLabels = Set[str]
                id_sets: Set[Tuple[Ids, DefaultLabels]] = set()
                for id_, default_label in id_to_label.items():
                    most_similar_id_set = None
                    best_score = 0
                    for id_set in id_sets:
                        sim = max(
                            self.spacy_pipeline.nlp(default_label.lower()).similarity(
                                self.spacy_pipeline.nlp(other_label.lower())
                            )
                            for other_label in id_set[1]
                        )
                        if sim > self.synonym_merge_threshold and sim > best_score:
                            most_similar_id_set = id_set

                    # for the first label, the above for loop is a no-op as id_sets is empty
                    # and the below if statement will be true.
                    # After that, it will be True if the id under consideration should not
                    # merge with any existing group and should get its own EquivalentIdSet
                    if not most_similar_id_set:
                        id_sets.add(
                            (
                                set((id_,)),
                                set((default_label,)),
                            )
                        )
                    else:
                        most_similar_id_set[0].add(id_)
                        most_similar_id_set[1].add(default_label)

                return frozenset(
                    EquivalentIdSet(
                        ids=frozenset(ids),
                        aggregated_by=EquivalentIdAggregationStrategy.RESOLVED_BY_SIMILARITY,
                        ids_to_source={id_: id_to_source[id_] for id_ in ids},
                    )
                    for ids, _ in id_sets
                )

    def _parse_df_if_not_already_parsed(self):
        if self.parsed_dataframe is None:
            self.parsed_dataframe = self.parse_to_dataframe()
            self.parsed_dataframe[DATA_ORIGIN] = self.data_origin
            self.parsed_dataframe[IDX] = self.parsed_dataframe[IDX].astype(str)
            self.parsed_dataframe.loc[
                pd.isnull(self.parsed_dataframe[DEFAULT_LABEL]), DEFAULT_LABEL
            ] = self.parsed_dataframe[IDX]

    def export_metadata(self) -> Dict[str, Dict[str, SimpleValue]]:
        self._parse_df_if_not_already_parsed()
        assert isinstance(self.parsed_dataframe, pd.DataFrame)
        metadata_columns = self.parsed_dataframe.columns.tolist()
        metadata_columns.remove(MAPPING_TYPE)
        metadata_columns.remove(SYN)
        metadata_df = self.parsed_dataframe[metadata_columns]
        metadata_df = metadata_df.drop_duplicates(subset=[IDX]).dropna(axis=0)
        metadata_df.set_index(inplace=True, drop=True, keys=IDX)
        assert set(OntologyParser.minimum_metadata_column_names).issubset(metadata_df.columns)
        metadata = metadata_df.to_dict(orient="index")
        return metadata

    def export_synonym_terms(self) -> Set[SynonymTerm]:
        self._parse_df_if_not_already_parsed()
        assert isinstance(self.parsed_dataframe, pd.DataFrame)
        # ensure correct order
        syn_df = self.parsed_dataframe[self.all_synonym_column_names].copy()
        syn_df = syn_df.dropna(subset=[SYN])
        syn_df[SYN] = syn_df[SYN].apply(str.strip)
        syn_df.drop_duplicates(subset=self.all_synonym_column_names)
        assert set(OntologyParser.all_synonym_column_names).issubset(syn_df.columns)
        synonym_terms = self.resolve_synonyms(synonym_df=syn_df)
        return synonym_terms

    def populate_metadata_database(self):
        """
        populate the metadata database with this ontology
        :return:
        """
        MetadataDatabase().add(self.name, self.export_metadata())

    def generate_synonyms(self) -> Set[SynonymTerm]:
        """
        generate synonyms based on configured synonym generator
        :return:
        """
        synonym_data = self.export_synonym_terms()
        generated_synonym_data = set()
        if self.synonym_generator:
            generated_synonym_data = self.synonym_generator(synonym_data)
        logger.info(
            f"{len(synonym_data)} original synonyms and {len(generated_synonym_data)} generated synonyms produced"
        )
        return generated_synonym_data

    def populate_synonym_database(self):
        """
        populate the synonym database
        :return:
        """

        SynonymDatabase().add(self.name, self.export_synonym_terms())

    def populate_databases(self):
        """
        populate the databases with the results of the parser
        """
        # populate the databases
        self.populate_metadata_database()
        self.populate_synonym_database()
        self.parsed_dataframe = None  # clear the reference to save memory

    def parse_to_dataframe(self) -> pd.DataFrame:
        """
        implementations should override this method, returning a 'long, thin' pd.DataFrame of at least the following
        columns:


        [IDX, DEFAULT_LABEL,DATA_ORIGIN, SYN, MAPPING_TYPE]

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
        raise NotImplementedError()

    def select_pos_pairs(self, df: pd.Series):
        """
        select synonym pair combinations for alignment. Capped at 50 to prevent overfitting
        :param df:
        :return:
        """
        raise NotImplementedError()

    def write_training_pairs(self, out_path: str):
        """
        write training pairs to a directory.
        :param out_path: directory to write to
        :return:
        """
        raise NotImplementedError()


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
        structure of the Ontology Parser superclass - i.e. should have keys for SYN, MAPPING_TYPE, DEFAULT_LABEL and
        IDX. All other keys are used as mapping metadata
        :param jsons_gen: iterator of python dict representing json objects
        :return: pd.DataFrame
        """
        raise NotImplementedError()


class OpenTargetsDiseaseOntologyParser(JsonLinesOntologyParser):
    name = "OPENTARGETS_DISEASE"
    allowed_sources = {"OGMS", "FBbt", "MONDO", "Orphanet", "EFO", "OTAR", "HP"}

    def find_kb(self, string: str) -> str:
        return string.split("_")[0]

    def json_dict_to_parser_dataframe(
        self, jsons_gen: Iterable[Dict[str, Any]]
    ) -> Iterable[pd.DataFrame]:
        # we ignore related syns for now until we decide how the system should handle them
        for json_dict in jsons_gen:
            idx = self.look_for_mondo(json_dict["id"], json_dict.get("dbXRefs", []))
            if any(allowed_source in idx for allowed_source in self.allowed_sources):
                synonyms = json_dict.get("synonyms", {})
                exact_syns = synonyms.get("hasExactSynonym", [])
                exact_syns.append(json_dict["name"])
                df = pd.DataFrame(exact_syns, columns=[SYN])
                df[MAPPING_TYPE] = "hasExactSynonym"
                df[DEFAULT_LABEL] = json_dict["name"]
                df[IDX] = idx
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

    annotation_fields = {
        "subcellularLocations",
        "tractability",
        "constraint",
        "functionDescriptions",
        "go",
        "hallmarks",
        "chemicalProbes",
        "safetyLiabilities",
        "pathways",
        "targetClass",
    }

    def score_and_group_ids(
        self,
        ids: Set[str],
        id_to_source: Dict[str, str],
        is_symbolic: bool,
        original_syn_set: Set[str],
    ) -> FrozenSet[EquivalentIdSet]:
        """
        custom score to deal with highly confused synonyms in opentargets
        in the case of a confused synonym, we pick the ID with the highest annotation score



        :param ids:
        :param id_to_source:
        :param is_symbolic:
        :return:
        """

        if not isinstance(self.spacy_pipeline, SpacyPipeline):
            associated_id_sets = frozenset(
                [
                    EquivalentIdSet(
                        ids=frozenset(ids),
                        aggregated_by=EquivalentIdAggregationStrategy.NO_STRATEGY,
                        ids_to_source={idx: id_to_source[idx] for idx in ids},
                    )
                ]
            )
            return associated_id_sets
        else:
            if len(ids) == 1:
                id_set = EquivalentIdSet(
                    ids=frozenset(ids),
                    aggregated_by=EquivalentIdAggregationStrategy.UNAMBIGUOUS,
                    ids_to_source={idx: id_to_source[idx] for idx in ids},
                )
                associated_id_sets = frozenset([id_set])
                return associated_id_sets
            else:
                id_to_label = {}
                label_to_id = defaultdict(set)
                id_to_score = {}

                for idx in ids:
                    approved_name: str = str(
                        self.metadata_db.get_by_idx(self.name, idx)["approvedName"]
                    )
                    annotation_score: int = int(
                        self.metadata_db.get_by_idx(self.name, idx)["annotation_score"]
                    )
                    id_to_label[idx] = approved_name
                    id_to_score[idx] = annotation_score
                    label_to_id[approved_name].add(idx)

                curated_ids = {idx for idx, score in id_to_score.items() if score > 0}
                if len(curated_ids) == 1:
                    # if we have info on the curated IDs, pick those, as it's more likely someone's hand curated it.
                    associated_id_sets = frozenset(
                        [
                            EquivalentIdSet(
                                ids=frozenset(curated_ids),
                                aggregated_by=EquivalentIdAggregationStrategy.CUSTOM,
                                ids_to_source={idx: id_to_source[idx] for idx in curated_ids},
                            )
                        ]
                    )
                    return associated_id_sets

                # if we have info on the curated IDs, pick those, as it's more likely someone's hand curated it.
                # if not, assume all are valid
                if len(curated_ids) > 1:
                    chosen_ids = curated_ids
                    label_to_id_subset = {
                        k: v for k, v in label_to_id.items() if len(v.intersection(chosen_ids)) > 0
                    }
                else:
                    chosen_ids = ids
                    label_to_id_subset = dict(label_to_id)

                if is_symbolic:
                    #     if is symbolic, it's very hard to know what to do. Symbols vary massively, so more or less
                    # impossible to determine which (if any) is correct
                    # if we have info on the curated IDs, pick those, as it's more likely someone's hand curated it.
                    # if not, assume all are valid

                    associated_id_sets = frozenset(
                        [
                            EquivalentIdSet(
                                ids=frozenset([idx]),
                                aggregated_by=EquivalentIdAggregationStrategy.SYNONYM_IS_AMBIGUOUS,
                                ids_to_source={idx: id_to_source[idx]},
                            )
                            for idx in chosen_ids
                        ]
                    )
                    return associated_id_sets

                else:
                    # for non symbolic ids, pick one based on similarity to the default label, if above a threshold.
                    # Otherwise, just assume the synonym is noisy

                    if len(original_syn_set) > 1:

                        # sorted call is so we consistently choose the same element for reproducibility
                        original_syn = next(iter(sorted(original_syn_set)))
                        logger.warning(
                            f"normaliser has merged two or more strings: {original_syn_set}. Chosen {original_syn}"
                            f"for similarity comparison"
                        )
                    else:
                        original_syn = next(iter(original_syn_set))
                    # if the normaliser hasn't merged two similar strings,

                    scores = []
                    for s1, s2 in itertools.product(label_to_id_subset.keys(), [original_syn]):
                        score = self.spacy_pipeline.nlp(s1.lower()).similarity(
                            self.spacy_pipeline.nlp(s2.lower())
                        )
                        scores.append((s1, score))
                    best_label, best_score = max(scores, key=lambda x: x[1])
                    if best_score <= self.synonym_merge_threshold:
                        # not similar enough to justify choosing one over another, so we keep all

                        associated_id_sets = frozenset(
                            [
                                EquivalentIdSet(
                                    ids=frozenset([idx]),
                                    aggregated_by=EquivalentIdAggregationStrategy.SYNONYM_IS_AMBIGUOUS,
                                    ids_to_source={idx: id_to_source[idx]},
                                )
                                for idx in chosen_ids
                            ]
                        )
                    else:
                        best_id = next(iter(label_to_id_subset[best_label]))
                        associated_id_sets = frozenset(
                            [
                                EquivalentIdSet(
                                    ids=frozenset([best_id]),
                                    aggregated_by=EquivalentIdAggregationStrategy.RESOLVED_BY_SIMILARITY,
                                    ids_to_source={best_id: id_to_source[best_id]},
                                )
                            ]
                        )
                    return associated_id_sets

    def find_kb(self, string: str) -> str:
        return "ENSEMBL"

    def json_dict_to_parser_dataframe(
        self, jsons_gen: Iterable[Dict[str, Any]]
    ) -> Iterable[pd.DataFrame]:
        for json_dict in jsons_gen:
            # due to a bug in OT data, TEC genes have "gene" as a synonym. Sunce they're uninteresting, we just filter
            # them
            biotype = json_dict.get("biotype")
            if biotype == "" or biotype == "tec" or json_dict["id"] == json_dict["approvedSymbol"]:
                continue
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

            annotation_score = sum(
                1
                for annotation_field in self.annotation_fields
                if len(json_dict.get(annotation_field, [])) > 0
            )

            records.append({SYN: json_dict["approvedSymbol"], MAPPING_TYPE: "approvedSymbol"})
            records.append({SYN: json_dict["approvedName"], MAPPING_TYPE: "approvedName"})
            records.append({SYN: json_dict["id"], MAPPING_TYPE: "opentargets_id"})
            df = pd.DataFrame.from_records(records, columns=[SYN, MAPPING_TYPE])
            df[IDX] = json_dict["id"]
            df[DEFAULT_LABEL] = json_dict["approvedSymbol"]
            df["dbXRefs"] = [json_dict.get("dbXRefs", [])] * df.shape[0]
            df["annotation_score"] = [annotation_score] * df.shape[0]
            df["approvedName"] = json_dict["approvedName"]
            yield df


class OpenTargetsMoleculeOntologyParser(JsonLinesOntologyParser):
    name = "OPENTARGETS_MOLECULE"

    def find_kb(self, string: str) -> str:
        return "CHEMBL"

    def json_dict_to_parser_dataframe(
        self, jsons_gen: Iterable[Dict[str, Any]]
    ) -> Iterable[pd.DataFrame]:
        for json_dict in jsons_gen:
            synonyms = json_dict.get("synonyms", [])
            main_name = json_dict["name"]
            synonyms.append(main_name)
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


class GeneOntologyParser(OntologyParser):
    name = "UNDEFINED"
    _uri_regex = re.compile("^http://purl.obolibrary.org/obo/GO_[0-9]+$")
    query = """UNDEFINED"""

    def load_go(self):
        g = rdflib.Graph()
        g.parse(self.in_path)
        return g

    def find_kb(self, string: str) -> str:
        return self.name

    def parse_to_dataframe(self) -> pd.DataFrame:
        g = rdflib.Graph()
        g.parse(self.in_path)
        result = g.query(self.query)
        default_labels = []
        iris = []
        syns = []
        mapping_type = []

        for row in result:
            idx = row.goid
            if "obsolete" in row.label:
                logger.info(f"skipping obsolete id: {row.goid}, {row.label}")
                continue
            if self._uri_regex.match(idx):
                default_labels.append(row.label)
                iris.append(row.goid)
                syns.append(row.synonym)
                mapping_type.append("hasExactSynonym")
        df = pd.DataFrame.from_dict(
            {DEFAULT_LABEL: default_labels, IDX: iris, SYN: syns, MAPPING_TYPE: mapping_type}
        )
        default_labels_df = df[[IDX, DEFAULT_LABEL]].drop_duplicates().copy()
        default_labels_df[SYN] = default_labels_df[DEFAULT_LABEL]
        default_labels_df[MAPPING_TYPE] = "label"

        return pd.concat([df, default_labels_df])


class BiologicalProcessGeneOntologyParser(GeneOntologyParser):
    name = "BP_GENE_ONTOLOGY"
    query = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX oboinowl: <http://www.geneontology.org/formats/oboInOwl#>

                SELECT DISTINCT ?goid ?label ?synonym
                        WHERE {

                            ?goid oboinowl:hasExactSynonym ?synonym .
                            ?goid rdfs:label ?label .
                            ?goid oboinowl:hasOBONamespace "biological_process" .

                  }
        """


class MolecularFunctionGeneOntologyParser(GeneOntologyParser):
    name = "MF_GENE_ONTOLOGY"
    query = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX oboinowl: <http://www.geneontology.org/formats/oboInOwl#>

                SELECT DISTINCT ?goid ?label ?synonym
                        WHERE {

                            ?goid oboinowl:hasExactSynonym ?synonym .
                            ?goid rdfs:label ?label .
                            ?goid oboinowl:hasOBONamespace "molecular_function".

                  }
        """


class CellularComponentGeneOntologyParser(GeneOntologyParser):
    name = "CC_GENE_ONTOLOGY"
    query = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX oboinowl: <http://www.geneontology.org/formats/oboInOwl#>

                SELECT DISTINCT ?goid ?label ?synonym
                        WHERE {

                            ?goid oboinowl:hasExactSynonym ?synonym .
                            ?goid rdfs:label ?label .
                            ?goid oboinowl:hasOBONamespace "cellular_component" .

                  }
        """


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

    def find_kb(self, string: str) -> str:
        return "UBERON"


class MondoOntologyParser(OntologyParser):
    name = "MONDO"
    _uri_regex = re.compile("^http://purl.obolibrary.org/obo/(MONDO|HP)_[0-9]+$")
    """
    input should be an MONDO json file
    e.g.
    https://www.ebi.ac.uk/ols/ontologies/mondo
    """

    def find_kb(self, string: str):
        path = parse.urlparse(string).path

        return path.split("/")[-1]

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
            if default_label is None:
                # skip if no default label is available
                continue
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

    def find_kb(self, string: str) -> str:
        return "ENSEMBL"

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

    def find_kb(self, string: str) -> str:
        return "CHEMBL"

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

    def find_kb(self, string: str) -> str:
        return "CLO"

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

    def find_kb(self, string: str) -> str:
        return "CELLOSAURUS"

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
    name = "MEDDRA_DISEASE"
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

    _exclude_soc = ["Surgical and medical procedures", "Social circumstances", "Investigations"]

    def find_kb(self, string: str) -> str:
        return "MEDDRA"

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
        hier_df = hier_df[~hier_df["soc_name"].isin(self._exclude_soc)]

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
