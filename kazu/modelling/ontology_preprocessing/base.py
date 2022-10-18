import json
import logging
import os
import re
import sqlite3
from abc import ABC, abstractmethod
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
from kazu.modelling.language.string_similarity_scorers import StringSimilarityScorer
from kazu.modelling.ontology_preprocessing.synonym_generation import CombinatorialSynonymGenerator
from kazu.utils.string_normalizer import StringNormalizer

DEFAULT_LABEL = "default_label"
IDX = "idx"
SYN = "syn"
MAPPING_TYPE = "mapping_type"
SOURCE = "source"
DATA_ORIGIN = "data_origin"


logger = logging.getLogger(__name__)


class OntologyParser(ABC):
    """
    Parse an ontology (or similar) into a set of outputs suitable for NLP entity linking.
    Implementations should have a class attribute 'name' to something suitably representative.
    The key method is parse_to_dataframe, which should convert an input source to a dataframe suitable
    for further processing.

    The other important method is find_kb. This should parse an ID string (if required) and return the underlying
    source. This is important for composite resources that contain identifiers from different seed sources
    """

    name = "unnamed"  # a label for this parser
    # the synonym table should have these (and only these columns)
    all_synonym_column_names = [IDX, SYN, MAPPING_TYPE]
    # the metadata table should have at least these columns (note, IDX will become the index)
    minimum_metadata_column_names = [DEFAULT_LABEL, DATA_ORIGIN]

    def __init__(
        self,
        in_path: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        entity_class: Optional[str] = None,
    ):
        """
        :param in_path: Path to some resource that should be processed (e.g. owl file, db config, tsv etc)
        :param string_scorer: Optional protocol of StringSimilarityScorer.  Used for resolving ambiguous symbolic
            synonyms via similarity calculation of the default label associated with the conflicted labels. If no
            instance is provided, all synonym conflicts will be assumed to refer to different concepts. This is not
            recommended!
        :param synonym_merge_threshold: similarity threshold to trigger a merge of conflicted synonyms into a single
            EquivalentIdSet. See docs for score_and_group_ids for further details
        :param data_origin: The origin of this dataset - e.g. HGNC release 2.1, MEDDRA 24.1 etc. Note, this is different
            from the parser.name, as is used to identify the origin of a mapping back to a data source
        :param synonym_generator: optional CombinatorialSynonymGenerator. Used to generate synonyms for dictionary
            based NER matching
        :param entity_class: optional str. the entity class associated with this parser, to pass to StringNormalizer.

            Generally speaking, when parsing a data source, synonyms that are
            symbolic (as determined by the StringNormalizer) that refer to more than one id are more likely to be
            ambiguous. Therefore, we assume they refer to unique concepts (e.g. COX 1 could be 'ENSG00000095303' OR
            'ENSG00000198804', and thus they will yield multiple instances of EquivalentIdSet.
            Non symbolic synonyms (i.e. noun phrases) are far less likely to refer to distinct entities, so we might
            want to merge the associated ID's non-symbolic ambiguous synonyms into a single EquivalentIdSet.
            The result of StringNormalizer.is_symbolic forms the is_symbolic parameter to .score_and_group_ids.

            If the underlying knowledgebase contains more than one entity type, muliple parsers should be
            implemented, subsetting accordingly (e.g. MEDDRA_DISEASE, MEDDRA_DIAGNOSTIC)

        """
        if string_scorer is None:
            logger.warning("no string scorer configured. Synonym resolution disabled.")
        self.string_scorer = string_scorer
        self.synonym_merge_threshold = synonym_merge_threshold
        self.data_origin = data_origin
        self.synonym_generator = synonym_generator
        self.in_path = in_path
        self.parsed_dataframe: Optional[pd.DataFrame] = None
        self.entity_class = entity_class
        self.metadata_db = MetadataDatabase()

    def find_kb(self, string: str) -> str:
        """
        split an IDX somehow to find the ontology SOURCE reference

        :param string: the IDX string to process
        :return:
        """
        raise NotImplementedError()

    def resolve_synonyms(self, synonym_df: pd.DataFrame) -> Set[SynonymTerm]:

        result = set()
        synonym_df["syn_norm"] = synonym_df[SYN].apply(
            StringNormalizer.normalize, entity_class=self.entity_class
        )

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

            is_symbolic = all(
                StringNormalizer.classify_symbolic(x, self.entity_class) for x in syn_set
            )

            ids: Set[str] = row[IDX]
            id_to_source = {}
            ontologies = set()
            for idx in ids:
                source = self.find_kb(idx)
                ontologies.add(source)
                id_to_source[idx] = source

            associated_id_sets, agg_strategy = self.score_and_group_ids(
                ids, id_to_source, is_symbolic, syn_set
            )

            synonym_term = SynonymTerm(
                term_norm=syn_norm,
                terms=frozenset(syn_set),
                is_symbolic=is_symbolic,
                mapping_types=mapping_type_set,
                associated_id_sets=associated_id_sets,
                parser_name=self.name,
                aggregated_by=agg_strategy,
            )

            result.add(synonym_term)

        return result

    def score_and_group_ids(
        self,
        ids: Set[str],
        id_to_source: Dict[str, str],
        is_symbolic: bool,
        original_syn_set: Set[str],
    ) -> Tuple[FrozenSet[EquivalentIdSet], EquivalentIdAggregationStrategy]:
        """
        for a given data source, one normalised synonym may map to one or more id. In some cases, the ID may be
        duplicate/redundant (e.g. there are many chembl ids for paracetamol). In other cases, the ID may refer to
        distinct concepts (e.g. COX 1 could be 'ENSG00000095303' OR 'ENSG00000198804').


        Since synonyms from data sources are confused in such a manner, we need to decide some way to cluster them into
        a single SynonymTerm concept, which in turn is a container for one or more EquivalentIdSet (depending on
        whether the concept is ambiguous or not)

        The job of score_and_group_ids is to determine how many EquivalentIdSet's for a given set of ids should be
        produced.

        The default algorithm (which can be overridden by concrete parser implementations) works as follows:

        1. If no StringScorer is configured, create an EquivalentIdSet for each id (strategy NO_STRATEGY -
           not recommended)
        2. If only one ID is referenced, or the associated normalised synonym string is not symbolic, group the
           ids into a single EquivalentIdSet (strategy UNAMBIGUOUS)
        3. otherwise, compare the default label associated with each ID to every other default label. If it's above
           self.synonym_merge_threshold, merge into one EquivalentIdSet, if not, create a new one

        recommendation: Use the SapbertStringSimilarityScorer for comparison

        IMPORTANT NOTE: any calls to this method requires the metadata DB to be populated, as this is the store of
        DEFAULT_LABEL

        :param ids: set of ids to decide upon
        :param id_to_source: mapping of id to original source
        :param is_symbolic: is the underlying synonym symbolic?
        :param original_syn_set: original synonyms associated with ids
        :return:
        """
        if self.string_scorer is None:
            # the NO_STRATEGY aggregation strategy assumes all synonyms are ambiguous
            return (
                frozenset(
                    EquivalentIdSet(
                        ids=frozenset((id_,)),
                        ids_to_source={id_: id_to_source[id_]},
                    )
                    for id_ in ids
                ),
                EquivalentIdAggregationStrategy.NO_STRATEGY,
            )
        else:

            if len(ids) == 1:
                return (
                    frozenset(
                        (
                            EquivalentIdSet(
                                ids=frozenset(ids),
                                ids_to_source={idx: id_to_source[idx] for idx in ids},
                            ),
                        )
                    ),
                    EquivalentIdAggregationStrategy.UNAMBIGUOUS,
                )

            id_to_label = {
                idx: str(self.metadata_db.get_by_idx(self.name, idx)[DEFAULT_LABEL]) for idx in ids
            }

            if not is_symbolic:
                return (
                    frozenset(
                        (
                            EquivalentIdSet(
                                ids=frozenset(ids),
                                ids_to_source={idx: id_to_source[idx] for idx in ids},
                            ),
                        )
                    ),
                    EquivalentIdAggregationStrategy.MERGED_AS_NON_SYMBOLIC,
                )
            else:
                # use similarity to group ids into EquivalentIdSets
                Ids = Set[str]
                DefaultLabels = Set[str]
                id_list: List[Tuple[Ids, DefaultLabels]] = []
                for id_, default_label in id_to_label.items():
                    most_similar_id_set = None
                    best_score = 0.0
                    for id_and_default_label_set in id_list:
                        sim = max(
                            self.string_scorer(default_label, other_label)
                            for other_label in id_and_default_label_set[1]
                        )
                        if sim > self.synonym_merge_threshold and sim > best_score:
                            most_similar_id_set = id_and_default_label_set

                    # for the first label, the above for loop is a no-op as id_sets is empty
                    # and the below if statement will be true.
                    # After that, it will be True if the id under consideration should not
                    # merge with any existing group and should get its own EquivalentIdSet
                    if not most_similar_id_set:
                        id_list.append(
                            (
                                {id_},
                                {default_label},
                            )
                        )
                    else:
                        most_similar_id_set[0].add(id_)
                        most_similar_id_set[1].add(default_label)

                return (
                    frozenset(
                        EquivalentIdSet(
                            ids=frozenset(ids),
                            ids_to_source={id_: id_to_source[id_] for id_ in ids},
                        )
                        for ids, _ in id_list
                    ),
                    EquivalentIdAggregationStrategy.RESOLVED_BY_SIMILARITY,
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
        metadata_columns = self.parsed_dataframe.columns
        metadata_columns.drop([MAPPING_TYPE, SYN])
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
        """
        MetadataDatabase().add_parser(self.name, self.export_metadata())

    def generate_synonyms(self) -> Set[SynonymTerm]:
        """
        generate synonyms based on configured synonym generator. Note, this method also calls
        populate_databases(), as the metadata db must be populated for appropriate synonym resolution
        """
        self.populate_databases()
        synonym_data = set(SynonymDatabase().get_all(self.name).values())
        generated_synonym_data = set()
        if self.synonym_generator:
            generated_synonym_data = self.synonym_generator(synonym_data)
        generated_synonym_data.update(synonym_data)
        logger.info(
            f"{len(synonym_data)} original synonyms and {len(generated_synonym_data)} generated synonyms produced"
        )
        return generated_synonym_data

    def populate_synonym_database(self):
        """
        populate the synonym database
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


        [IDX, DEFAULT_LABEL, SYN, MAPPING_TYPE]

        IDX: the ontology id
        DEFAULT_LABEL: the preferred label
        SYN: a synonym of the concept
        MAPPING_TYPE: the type of mapping from default label to synonym - e.g. xref, exactSyn etc. Usually defined by the ontology

        Note: It is the responsibility of the implementation of parse_to_dataframe to add default labels as synonyms.
        """
        raise NotImplementedError()

    def format_training_table(self) -> pd.DataFrame:
        """
        generate a table of synonym pairs. Useful for aligning an embedding space (e.g. as for sapbert)
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
        :return:
        """
        raise NotImplementedError()


class OpenTargetsDiseaseOntologyParser(JsonLinesOntologyParser):
    name = "OPENTARGETS_DISEASE"
    # Just use IDs that are in MONDO, since that's all people in general care about.
    # if we want to expand this out, other sources are:
    # "OGMS", "FBbt", "Orphanet", "EFO", "OTAR"
    # but we did have these in previously, and EFO introduced a lot of noise as it
    # has non-disease terms like 'dose' that occur frequently.
    # we could make the allowed sources a config option but we don't need to configure
    # currently, and easy to change later (and provide the current value as a default if
    # not present in config)
    allowed_sources = {"MONDO", "HP"}

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
    ) -> Tuple[FrozenSet[EquivalentIdSet], EquivalentIdAggregationStrategy]:
        """
        since non symbolic gene symbols are also frequently ambiguous, we override this method accordingly to disable
        all synonym resolution, and rely on disambiguation to decide on 'true' mappings. Answers on a postcard if anyone
        has a better idea on how to do this!

        :param ids:
        :param id_to_source:
        :param is_symbolic:
        :param original_syn_set:
        :return:
        """

        return (
            frozenset(
                EquivalentIdSet(
                    ids=frozenset((id_,)),
                    ids_to_source={id_: id_to_source[id_]},
                )
                for id_ in ids
            ),
            EquivalentIdAggregationStrategy.CUSTOM,
        )

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
        """
        raise NotImplementedError()

    def parse_to_dataframe(self) -> pd.DataFrame:
        g = rdflib.Graph()
        g.parse(self.in_path)
        label_pred_str = "http://www.w3.org/2000/01/rdf-schema#label"
        label_predicates = rdflib.URIRef(label_pred_str)
        synonym_predicates = [rdflib.URIRef(x) for x in self._get_synonym_predicates()]
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

        Uses `self._uri_regex` to define valid IRIs
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
    input should be a MONDO json file
    e.g.
    https://www.ebi.ac.uk/ols/ontologies/mondo
    """

    def find_kb(self, string: str):
        path = parse.urlparse(string).path
        # just the final bit, e.g. MONDO_0000123
        path_end = path.split("/")[-1]
        # we don't want the underscore or digits for the unique ID, just the ontology bit
        return path_end.split("_")[0]

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
    """
    input is a sqllite dump from Chembl, e.g.
    https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_29_sqlite.tar.gz
    """

    name = "CHEMBL"

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
    """
    input is an obo file from cellosaurus, e.g.
    https://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo
    """

    name = "CELLOSAURUS"
    cell_line_re = re.compile("cell line", re.IGNORECASE)

    def find_kb(self, string: str) -> str:
        return "CELLOSAURUS"

    def score_and_group_ids(
        self,
        ids: Set[str],
        id_to_source: Dict[str, str],
        is_symbolic: bool,
        original_syn_set: Set[str],
    ) -> Tuple[FrozenSet[EquivalentIdSet], EquivalentIdAggregationStrategy]:
        """
        treat all synonyms as seperate cell lines

        :param ids:
        :param id_to_source:
        :param is_symbolic:
        :param original_syn_set:
        :return:
        """

        return (
            frozenset(
                EquivalentIdSet(
                    ids=frozenset((id_,)),
                    ids_to_source={id_: id_to_source[id_]},
                )
                for id_ in ids
            ),
            EquivalentIdAggregationStrategy.CUSTOM,
        )

    def _remove_cell_line_text(self, text: str):
        return self.cell_line_re.sub("", text).strip()

    def parse_to_dataframe(self) -> pd.DataFrame:

        ids = []
        default_labels = []
        all_syns = []
        mapping_type = []
        with open(self.in_path, "r") as f:
            id = ""
            for line in f:
                text = line.rstrip()
                if text.startswith("id:"):
                    id = text.split(" ")[1]
                elif text.startswith("name:"):
                    default_label = text[5:].strip()
                    ids.append(id)
                    # we remove "cell line" because they're all cell lines and it confuses mapping
                    default_label_no_cell_line = self._remove_cell_line_text(default_label)
                    default_labels.append(default_label_no_cell_line)
                    all_syns.append(default_label_no_cell_line)
                    mapping_type.append("name")
                # synonyms in cellosaurus are a bit of a mess, so we don't use this field for now. Leaving this here
                # in case they improve at some point
                # elif text.startswith("synonym:"):
                #     match = self._synonym_regex.match(text)
                #     if match is None:
                #         raise ValueError(
                #             """synonym line does not match our synonym regex.
                #             Either something is wrong with the file, or it has updated
                #             and our regex is not correct/general enough."""
                #         )
                #     ids.append(id)
                #     default_labels.append(default_label)
                #
                #     all_syns.append(self._remove_cell_line_text(match.group("syn")))
                #     mapping_type.append(match.group("mapping"))
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
    """
    input is an unzipped directory to a MEddra release (Note, requires licence). This
    should contain the files 'mdhier.asc' and 'llt.asc'
    """

    name = "MEDDRA_DISEASE"

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
