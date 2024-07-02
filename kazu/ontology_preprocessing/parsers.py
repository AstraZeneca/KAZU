"""This module consists entirely of implementations of :class:`~.OntologyParser`.

Some of these are aimed specifically at a custom format for individual
ontologies, like :class:`~.ChemblOntologyParser` or
:class:`~.MeddraOntologyParser`.

Others are aimed to provide flexibly for a user across a format, such as
:class:`~.RDFGraphParser`, :class:`~.TabularOntologyParser` and
:class:`~.JsonLinesOntologyParser`.

If you do not find a parser that meets your needs, please see
:ref:`writing-a-custom-parser`.
"""
import copy
import itertools
import json
import logging
import os
import re
import sqlite3
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import cast, Any, Optional, Union, overload
from collections.abc import Iterable
from urllib import parse

import pandas as pd
import rdflib
import packaging.version

from kazu.database.in_memory_db import MetadataDatabase

from kazu.data import (
    EquivalentIdSet,
    EquivalentIdAggregationStrategy,
    OntologyStringResource,
    AssociatedIdSets,
    GlobalParserActions,
)
from kazu.language.string_similarity_scorers import StringSimilarityScorer
from kazu.ontology_preprocessing.base import (
    OntologyParser,
    SYN,
    MAPPING_TYPE,
    DEFAULT_LABEL,
    IDX,
    IdsAndSource,
)
from kazu.ontology_preprocessing.synonym_generation import CombinatorialSynonymGenerator
from kazu.utils.grouping import sort_then_group
from kazu.ontology_preprocessing.autocuration import AutoCurator
from kazu.utils.utils import PathLike
from kazu.ontology_preprocessing.downloads import OntologyDownloader

logger = logging.getLogger(__name__)


class JsonLinesOntologyParser(OntologyParser):
    """A parser for a jsonlines dataset.

    Assumes one kb entry per line (i.e. json object).

    This should be subclassed and subclasses must implement
    :meth:`~.json_dict_to_parser_records`.
    """

    def read(self, path: Path) -> Iterable[dict[str, Any]]:
        for json_path in path.glob("*.json"):
            with json_path.open(mode="r") as f:
                for line in f:
                    yield json.loads(line)

    def parse_to_dataframe(self):
        return pd.DataFrame.from_records(self.json_dict_to_parser_records(self.read(self.in_path)))

    def json_dict_to_parser_records(
        self, jsons_gen: Iterable[dict[str, Any]]
    ) -> Iterable[dict[str, Any]]:
        """For a given input json (represented as a python dict), yield dictionary
        record(s) compatible with the expected structure of the Ontology Parser
        superclass.

        This means dictionaries should have keys for :data:`~.SYN`,
        :data:`~.MAPPING_TYPE`, :data:`~.DEFAULT_LABEL` and
        :data:`~.IDX`. All other keys are used as mapping metadata.

        :param jsons_gen: iterable of python dict representing json objects
        :return:
        """
        raise NotImplementedError


class OpenTargetsDiseaseOntologyParser(JsonLinesOntologyParser):
    """Parser for OpenTargets Disease release.

    OpenTargets has a lot of entities in its disease dataset, not all of which are diseases. Here,
    we use the ``allowed_therapeutic_areas`` argument to describe which specific therapeutic areas
    a given instance of this parser should use. See https://platform-docs.opentargets.org/disease-or-phenotype
    for more info.
    """

    DF_XREF_FIELD_NAME = "dbXRefs"

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        allowed_therapeutic_areas: Iterable[str],
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        ontology_downloader: Optional[OntologyDownloader] = None,
    ):
        """

        :param in_path:
        :param entity_class:
        :param name:
        :param allowed_therapeutic_areas: areas to use in this instance. These are IDs in OpenTargets' format, like ``MONDO_0024458``.
        :param string_scorer:
        :param synonym_merge_threshold:
        :param data_origin:
        :param synonym_generator:
        :param autocurator:
        :param curations_path:
        :param global_actions:
        :param: ontology_downloader:
        """
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            curations_path=curations_path,
            global_actions=global_actions,
            autocurator=autocurator,
            ontology_downloader=ontology_downloader,
        )
        self.allowed_therapeutic_areas = set(allowed_therapeutic_areas)
        self.metadata_db = MetadataDatabase()

    def find_kb(self, string: str) -> str:
        return string.split("_")[0]

    def score_and_group_ids(
        self,
        ids_and_source: IdsAndSource,
        is_symbolic: bool,
    ) -> tuple[AssociatedIdSets, EquivalentIdAggregationStrategy]:
        """Group disease IDs via cross-reference.

        Falls back to superclass implementation if any xrefs are inconsistently
        described.

        :param ids_and_source:
        :param is_symbolic:
        :return:
        """
        if len(ids_and_source) == 1:
            return super().score_and_group_ids(
                ids_and_source=ids_and_source, is_symbolic=is_symbolic
            )

        unmapped_ids_and_sources = copy.deepcopy(ids_and_source)

        # look up all cross-references in the DB
        xref_lookup = {}
        for idx_and_source in ids_and_source:
            xref_set = set(
                self.metadata_db.get_by_idx(name=self.name, idx=idx_and_source[0])[
                    self.DF_XREF_FIELD_NAME
                ]
            )
            # we also need to add in the OT default one, that needs some parsing as is in a slightly different
            # format
            xref_set.add(idx_and_source[0].replace("_", ":"))
            xref_lookup[idx_and_source] = xref_set

        # now do a pair-wise comparison of the xrefs attached to each ID, and map the idx_and_source
        # to the intersection
        groups = defaultdict(set)
        for (idx_and_source1, xref_set1), (idx_and_source2, xref_set2) in itertools.combinations(
            xref_lookup.items(), r=2
        ):
            matched_xrefs = frozenset(xref_set1.intersection(xref_set2))
            if len(matched_xrefs) > 0:
                groups[matched_xrefs].add(idx_and_source1)
                groups[matched_xrefs].add(idx_and_source2)
                unmapped_ids_and_sources.discard(idx_and_source1)
                unmapped_ids_and_sources.discard(idx_and_source2)

        if len(groups) > 1:
            for set_1, set_2 in itertools.combinations(groups.values(), r=2):
                if not set_1.isdisjoint(set_2):

                    # for this set of ids, xref mappings are confused between two or more subsets
                    # so fall back to default method
                    return super().score_and_group_ids(
                        ids_and_source=ids_and_source, is_symbolic=is_symbolic
                    )

        # now add in any remaining unmapped ids as separate groups
        groups_list = list(groups.values())
        for unmapped_id_and_source in unmapped_ids_and_sources:
            groups_list.append({unmapped_id_and_source})

        assoc_id_sets = frozenset(
            EquivalentIdSet(ids_and_source=frozenset(grouped_ids_and_source))
            for grouped_ids_and_source in groups_list
        )
        return assoc_id_sets, EquivalentIdAggregationStrategy.RESOLVED_BY_XREF

    def json_dict_to_parser_records(
        self, jsons_gen: Iterable[dict[str, Any]]
    ) -> Iterable[dict[str, Any]]:
        # we ignore related syns for now until we decide how the system should handle them
        for json_dict in jsons_gen:
            default_label: str = json_dict["name"]

            if set(json_dict.get("therapeuticAreas", ())).isdisjoint(
                self.allowed_therapeutic_areas
            ):
                logger.debug(
                    "skipping entry not included in allowed_therapeutic_areas: %s", json_dict
                )
                continue

            idx = json_dict["id"]
            dbXRefs = json_dict.get("dbXRefs", [])
            yield {
                SYN: default_label,
                MAPPING_TYPE: "name",
                DEFAULT_LABEL: default_label,
                IDX: idx,
                "dbXRefs": dbXRefs,
            }
            synonyms = json_dict.get("synonyms", {})
            exact_syns = synonyms.get("hasExactSynonym", [])
            for syn in exact_syns:
                yield {
                    SYN: syn,
                    MAPPING_TYPE: "hasExactSynonym",
                    DEFAULT_LABEL: default_label,
                    IDX: idx,
                    "dbXRefs": dbXRefs,
                }


class OpenTargetsTargetOntologyParser(JsonLinesOntologyParser):
    """Parser for the OT Target dataset.

    .. note:: Automatically ignored records

       Since there are many thousands of ensembl IDs that reference uninteresting
       genomic locations, we will likely never see them in natural language.
       Therefore, we automatically filter records that do not have an approved
       symbol defined. In addition, this class allows one to filter biotypes they're
       not interested in using the ``excluded_biotypes`` argument of the constructor.
    """

    ANNOTATION_FIELDS = {
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

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        excluded_biotypes: Optional[Iterable[str]] = None,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        ontology_downloader: Optional[OntologyDownloader] = None,
    ):
        """

        :param in_path:
        :param entity_class:
        :param name:
        :param excluded_biotypes: if specified, ignore these biotypes. Note that an empty string "" is a biotype in OT
            for some reason...
        :param string_scorer:
        :param synonym_merge_threshold:
        :param data_origin:
        :param synonym_generator:
        :param autocurator:
        :param curations_path:
        :param global_actions:
        :param ontology_downloader:
        """
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            curations_path=curations_path,
            global_actions=global_actions,
            autocurator=autocurator,
            ontology_downloader=ontology_downloader,
        )
        self.excluded_biotypes: set[str] = set(excluded_biotypes) if excluded_biotypes else set()

    def score_and_group_ids(
        self,
        ids_and_source: IdsAndSource,
        is_symbolic: bool,
    ) -> tuple[AssociatedIdSets, EquivalentIdAggregationStrategy]:
        """Group Ensembl gene IDs belonging to the same gene.

        .. admonition:: Note for non-biologists about genes
           :class: note

           The concept of a 'gene' is complex, and Ensembl gene IDs actually refer
           to locations on the genome, rather than individual genes. In fact, one
           'gene' can be made up of multiple Ensembl gene IDs, generally speaking
           these are exons that produce different isoforms of a given protein.

        :param ids_and_source:
        :param is_symbolic:
        :return:
        """
        meta_db = MetadataDatabase()
        assoc_id_sets: set[EquivalentIdSet] = set()

        for _, grouped_ids_and_source in sort_then_group(
            ids_and_source, lambda x: meta_db.get_by_idx(name=self.name, idx=x[0])[DEFAULT_LABEL]
        ):
            assoc_id_sets.add(
                EquivalentIdSet(
                    ids_and_source=frozenset(
                        (
                            grouped_idx,
                            source,
                        )
                        for grouped_idx, source in grouped_ids_and_source
                    )
                )
            )
        return frozenset(assoc_id_sets), EquivalentIdAggregationStrategy.CUSTOM

    def find_kb(self, string: str) -> str:
        return "ENSEMBL"

    def json_dict_to_parser_records(
        self, jsons_gen: Iterable[dict[str, Any]]
    ) -> Iterable[dict[str, Any]]:
        for json_dict in jsons_gen:
            if json_dict.get("biotype") in self.excluded_biotypes:
                continue

            # if no approved symbol is assigned, ignore the record
            if json_dict["id"] == json_dict["approvedSymbol"]:
                continue

            annotation_score = sum(
                1
                for annotation_field in self.ANNOTATION_FIELDS
                if len(json_dict.get(annotation_field, [])) > 0
            )

            idx = json_dict["id"]
            default_label = json_dict["approvedSymbol"]
            shared_values = {
                IDX: idx,
                DEFAULT_LABEL: default_label,
                "dbXRefs": json_dict.get("dbXRefs", []),
                "approvedName": json_dict["approvedName"],
                "annotation_score": annotation_score,
            }

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
                    record.update(shared_values)
                    yield record

            for key in ("approvedSymbol", "approvedName", "id"):
                if key == "id":
                    mapping_type = "opentargets_id"
                else:
                    mapping_type = key

                res = {SYN: json_dict[key], MAPPING_TYPE: mapping_type}
                res.update(shared_values)
                yield res


class OpenTargetsMoleculeOntologyParser(JsonLinesOntologyParser):
    def find_kb(self, string: str) -> str:
        return "CHEMBL"

    def json_dict_to_parser_records(
        self, jsons_gen: Iterable[dict[str, Any]]
    ) -> Iterable[dict[str, Any]]:
        for json_dict in jsons_gen:
            cross_references = json_dict.get("crossReferences", {})
            default_label = json_dict["name"]
            idx = json_dict["id"]

            synonyms = json_dict.get("synonyms", [])
            main_name = json_dict["name"]
            synonyms.append(main_name)

            for syn in synonyms:
                yield {
                    SYN: syn,
                    MAPPING_TYPE: "synonyms",
                    "crossReferences": cross_references,
                    DEFAULT_LABEL: default_label,
                    IDX: idx,
                }

            for trade_name in json_dict.get("tradeNames", []):
                yield {
                    SYN: trade_name,
                    MAPPING_TYPE: "tradeNames",
                    "crossReferences": cross_references,
                    DEFAULT_LABEL: default_label,
                    IDX: idx,
                }


RdfRef = Union[rdflib.paths.Path, rdflib.term.Node, str]
# Note - lists are actually normally provided here through hydra config
# but there's apparently no way of type hinting
# 'any iterable of length two where the items have these types'
PredicateAndValue = tuple[RdfRef, rdflib.term.Node]
EntityFilteringPatterns = Optional[Iterable[PredicateAndValue]]


class RDFGraphParser(OntologyParser):
    """Parser for rdf files.

    Supports any format of rdf that :meth:`rdflib.Graph.parse` can infer
    the format of from the file extension, e.g. ``.xml`` , ``.ttl`` ,
    ``.owl`` , ``.json``. Case of the extension does not matter. This
    functionality is handled by :func:`rdflib.util.guess_format`, but
    will fall back to attempting to parse as turtle/ttl format in the
    case of an unknown file extension.
    """

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        uri_regex: Union[str, re.Pattern],
        synonym_predicates: Iterable[RdfRef],
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.7,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        include_entity_patterns: EntityFilteringPatterns = None,
        exclude_entity_patterns: EntityFilteringPatterns = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        ontology_downloader: Optional[OntologyDownloader] = None,
        label_predicate: RdfRef = rdflib.RDFS.label,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            curations_path=curations_path,
            global_actions=global_actions,
            autocurator=autocurator,
            ontology_downloader=ontology_downloader,
        )

        if isinstance(uri_regex, re.Pattern):
            self._uri_regex = uri_regex
        else:
            self._uri_regex = re.compile(uri_regex)

        self.synonym_predicates = tuple(
            self.convert_to_rdflib_ref(pred) for pred in synonym_predicates
        )
        self.label_predicate = self.convert_to_rdflib_ref(label_predicate)

        if include_entity_patterns is not None:
            self.include_entity_patterns = tuple(
                (self.convert_to_rdflib_ref(pred), self.convert_to_rdflib_ref(val))
                for pred, val in include_entity_patterns
            )
        else:
            self.include_entity_patterns = tuple()

        if exclude_entity_patterns is not None:
            self.exclude_entity_patterns = tuple(
                (self.convert_to_rdflib_ref(pred), self.convert_to_rdflib_ref(val))
                for pred, val in exclude_entity_patterns
            )
        else:
            self.exclude_entity_patterns = tuple()

    def find_kb(self, string: str) -> str:
        """By default, just return the name of the parser.

        If more complex behaviour is necessary, write a custom subclass and override
        this method.
        """
        return self.name

    @overload
    @staticmethod
    def convert_to_rdflib_ref(pred: rdflib.paths.Path) -> rdflib.paths.Path:
        pass

    @overload
    @staticmethod
    def convert_to_rdflib_ref(pred: rdflib.term.Node) -> rdflib.term.Node:
        pass

    @overload
    @staticmethod
    def convert_to_rdflib_ref(pred: str) -> rdflib.URIRef:
        pass

    @staticmethod
    def convert_to_rdflib_ref(pred):
        if isinstance(pred, (rdflib.term.Node, rdflib.paths.Path)):
            return pred
        else:
            return rdflib.URIRef(pred)

    @staticmethod
    def parse_to_graph(in_path: Path) -> rdflib.Graph:
        """Parse the given input path using `rdflib <https://rdflib.readthedocs.io/en/stable/index.html>`_.

        Called by :meth:`~.RDFGraphParser.parse_to_dataframe`, this is a separate method to allow
        overriding to tweak the parsing process, such as adding caching
        (as in :meth:`~.GeneOntologyParser.parse_to_graph`).
        """
        return rdflib.Graph().parse(in_path)

    def parse_to_dataframe(self) -> pd.DataFrame:
        g = self.parse_to_graph(self.in_path)
        default_labels = []
        iris = []
        syns = []
        mapping_type = []

        label_pred_str = str(self.label_predicate)

        for sub, obj in g.subject_objects(self.label_predicate):
            if not self.is_valid_iri(str(sub)):
                continue

            if any((sub, pred, value) not in g for pred, value in self.include_entity_patterns):
                continue

            if any((sub, pred, value) in g for pred, value in self.exclude_entity_patterns):
                continue

            default_labels.append(str(obj))
            iris.append(str(sub))
            syns.append(str(obj))
            mapping_type.append(label_pred_str)
            for syn_predicate in self.synonym_predicates:
                for other_syn_obj in g.objects(subject=sub, predicate=syn_predicate):
                    default_labels.append(str(obj))
                    iris.append(str(sub))
                    syns.append(str(other_syn_obj))
                    mapping_type.append(str(syn_predicate))

        df = pd.DataFrame.from_dict(
            {DEFAULT_LABEL: default_labels, IDX: iris, SYN: syns, MAPPING_TYPE: mapping_type}
        )
        return df

    def is_valid_iri(self, text: str) -> bool:
        """Check if input string is a valid IRI for the ontology being parsed.

        Uses ``self._uri_regex`` to define valid IRIs.
        """
        match = self._uri_regex.match(text)
        return bool(match)


SKOS_XL_PREF_LABEL_PATH: rdflib.paths.Path = rdflib.URIRef(
    "http://www.w3.org/2008/05/skos-xl#prefLabel"
) / rdflib.URIRef("http://www.w3.org/2008/05/skos-xl#literalForm")
SKOS_XL_ALT_LABEL_PATH: rdflib.paths.Path = rdflib.URIRef(
    "http://www.w3.org/2008/05/skos-xl#altLabel"
) / rdflib.URIRef("http://www.w3.org/2008/05/skos-xl#literalForm")


class SKOSXLGraphParser(RDFGraphParser):
    """Parse SKOS-XL RDF Files.

    Note that this just sets a default label predicate and synonym predicate to SKOS-XL
    appropriate paths, and then passes to the parent RDFGraphParser class. This class is
    just a convenience to make specifying a SKOS-XL parser easier, this functionality is
    still available via RDFGraphParser directly.
    """

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        uri_regex: Union[str, re.Pattern],
        synonym_predicates: Iterable[RdfRef] = (SKOS_XL_ALT_LABEL_PATH,),
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.7,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        include_entity_patterns: EntityFilteringPatterns = None,
        exclude_entity_patterns: EntityFilteringPatterns = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        ontology_downloader: Optional[OntologyDownloader] = None,
        label_predicate: RdfRef = SKOS_XL_PREF_LABEL_PATH,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            uri_regex=uri_regex,
            synonym_predicates=synonym_predicates,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            include_entity_patterns=include_entity_patterns,
            exclude_entity_patterns=exclude_entity_patterns,
            curations_path=curations_path,
            global_actions=global_actions,
            label_predicate=label_predicate,
            autocurator=autocurator,
            ontology_downloader=ontology_downloader,
        )


class GeneOntologyParser(RDFGraphParser):
    """A parser for the `Gene Ontology <https://geneontology.org>`_.

    Differences from its parent class :class:`~.RDFGraphParser`:

    1. Specify an appropriate ``uri_regex`` and ``synonym_predicates`` for the Gene Ontology.
    2. Drop entities with a defalt label containing ``obsolete`` - see :meth:`~.parse_to_dataframe`.
    3. Cache the parsing of the rdf file, since we have multiple parsers that use different parts of the ontology,
       so this saves re-parsing the source file multiple times, which is a signficant cost as the file is very large
       and parsing rdf is expensive.

    Subclasses of this class like :class:`~.BiologicalProcessGeneOntologyParser` filter
    to a specific 'namespace' within the Gene Ontology. These are present as a convenience
    and for discoverability. It is straightforward to configure a ``GeneOntologyParser``
    instance to filter to a namespace without subclassing - see the implementation of
    :class:`~.BiologicalProcessGeneOntologyParser` for details."""

    instances: set[str] = set()
    instances_in_dbs: set[str] = set()

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        include_entity_patterns: EntityFilteringPatterns = None,
        exclude_entity_patterns: EntityFilteringPatterns = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        ontology_downloader: Optional[OntologyDownloader] = None,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            uri_regex=re.compile("^http://purl.obolibrary.org/obo/GO_[0-9]+$"),
            synonym_predicates=(
                rdflib.URIRef("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"),
            ),
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            include_entity_patterns=include_entity_patterns,
            exclude_entity_patterns=exclude_entity_patterns,
            curations_path=curations_path,
            global_actions=global_actions,
            autocurator=autocurator,
            ontology_downloader=ontology_downloader,
        )
        self.instances.add(name)

    def populate_databases(
        self, force: bool = False, return_resources: bool = False
    ) -> Optional[list[OntologyStringResource]]:
        """Modified version of :meth:`RDFGraphParser.parse_to_graph` to handle caching.

        We have custom logic here to clear the caching on :meth:`~.parse_to_graph`,
        because the size of this cached graph is quite large in memory, and otherwise stays
        in continued usage throughout the runtime of kazu.
        """
        resources = super().populate_databases(force=force, return_resources=return_resources)
        self.instances_in_dbs.add(self.name)

        if self.instances_in_dbs >= self.instances:
            # all existing instances are in the database, so we can free up
            # the memory used by the cached parsed gene ontology, which is significant.
            self.parse_to_graph.cache_clear()
        return resources

    @staticmethod
    @cache
    def parse_to_graph(in_path: str) -> rdflib.Graph:
        """Cached version of :meth:`RDFGraphParser.parse_to_graph`.

        Cached due to the expense of parsing Gene Ontology from scratch (otherwise we
        end up doing this 3 times in the public model pack).
        """
        # needs to provide explicit arguments because this is a staticmethod
        return super(GeneOntologyParser, GeneOntologyParser).parse_to_graph(in_path)

    def parse_to_dataframe(self) -> pd.DataFrame:
        """A modification of :meth:`RDFGraphParser.parse_to_dataframe`.

        The only difference from the overriden method is that this drops entities where
        the default label contains ``obsolete``, as these are no longer relevant for Gene
        Ontology NER/Entity Linking.
        """
        df = super().parse_to_dataframe()
        df = df.drop(df.index[df[DEFAULT_LABEL].str.contains("obsolete")])
        return df

    def __del__(self):
        GeneOntologyParser.instances.discard(self.name)


class BiologicalProcessGeneOntologyParser(GeneOntologyParser):
    """A subclass of :class:`GeneOntologyParser` that filters to only the
    ``biological_process`` namespace."""

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        ontology_downloader: Optional[OntologyDownloader] = None,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            name=name,
            curations_path=curations_path,
            global_actions=global_actions,
            include_entity_patterns=[
                (
                    rdflib.URIRef("http://www.geneontology.org/formats/oboInOwl#hasOBONamespace"),
                    rdflib.Literal("biological_process"),
                )
            ],
            exclude_entity_patterns=[],
            autocurator=autocurator,
            ontology_downloader=ontology_downloader,
        )


class MolecularFunctionGeneOntologyParser(GeneOntologyParser):
    """A subclass of :class:`GeneOntologyParser` that filters to only the
    ``molecular_function`` namespace."""

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        ontology_downloader: Optional[OntologyDownloader] = None,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            name=name,
            curations_path=curations_path,
            global_actions=global_actions,
            include_entity_patterns=[
                (
                    rdflib.URIRef("http://www.geneontology.org/formats/oboInOwl#hasOBONamespace"),
                    rdflib.Literal("molecular_function"),
                )
            ],
            exclude_entity_patterns=[],
            autocurator=autocurator,
            ontology_downloader=ontology_downloader,
        )


class CellularComponentGeneOntologyParser(GeneOntologyParser):
    """A subclass of :class:`GeneOntologyParser` that filters to only the
    ``cellular_component`` namespace."""

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        ontology_downloader: Optional[OntologyDownloader] = None,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            name=name,
            curations_path=curations_path,
            global_actions=global_actions,
            include_entity_patterns=[
                (
                    rdflib.URIRef("http://www.geneontology.org/formats/oboInOwl#hasOBONamespace"),
                    rdflib.Literal("cellular_component"),
                )
            ],
            exclude_entity_patterns=[],
            autocurator=autocurator,
            ontology_downloader=ontology_downloader,
        )


class UberonOntologyParser(RDFGraphParser):
    """Input should be an UBERON owl file e.g.
    https://www.ebi.ac.uk/ols/ontologies/uberon."""

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        ontology_downloader: Optional[OntologyDownloader] = None,
    ):

        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            uri_regex=re.compile("^http://purl.obolibrary.org/obo/UBERON_[0-9]+$"),
            synonym_predicates=(
                rdflib.URIRef("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"),
            ),
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            curations_path=curations_path,
            global_actions=global_actions,
            autocurator=autocurator,
            ontology_downloader=ontology_downloader,
        )

    def find_kb(self, string: str) -> str:
        return "UBERON"


class MondoOntologyParser(OntologyParser):
    _uri_regex = re.compile("^http://purl.obolibrary.org/obo/(MONDO|HP)_[0-9]+$")
    """Input should be a MONDO json file e.g.
    https://www.ebi.ac.uk/ols/ontologies/mondo."""

    def find_kb(self, string: str) -> str:
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


class HGNCGeneOntologyParser(OntologyParser):
    """Parse HGNC data and extract individual genes as entities.

    Input is a json from HGNC. For example,
    http://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/json/hgnc_complete_set.json.
    """

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        ontology_downloader: Optional[OntologyDownloader] = None,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            name=name,
            curations_path=curations_path,
            global_actions=global_actions,
            autocurator=autocurator,
            ontology_downloader=ontology_downloader,
        )

    def find_kb(self, string: str) -> str:
        return "ENSEMBL"

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
        all_mapping_type: list[str] = []
        docs = data["response"]["docs"]
        for doc in docs:

            def get_with_default_list(key: str) -> list[str]:
                found = doc.get(key, [])
                if not isinstance(found, list):
                    found = [found]
                return cast(list[str], found)

            ensembl_gene_id = doc.get("ensembl_gene_id", None)
            name = doc.get("name", None)
            if ensembl_gene_id is None or name is None:
                continue
            else:
                # find synonyms
                synonyms: list[tuple[str, str]] = []
                for hgnc_key in keys_to_check:
                    synonyms_this_entity = get_with_default_list(hgnc_key)
                    for potential_synonym in synonyms_this_entity:
                        synonyms.append((potential_synonym, hgnc_key))

                synonyms = list(set(synonyms))
                synonyms_strings = []
                for synonym_str, mapping_t in synonyms:
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
    """Input is a directory containing an extracted sqllite dump from Chembl.

    .. deprecated:: 2.1.0
       Use :class:`kazu.ontology_preprocessing.parsers.ChemblParquetOntologyParser` instead.
       This is deprecated so we don't have to store a large sqlite database file in resources.


    For example, this can be sourced from:
    https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_33/chembl_33_sqlite.tar.gz.
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
            SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, pref_name AS {SYN}, 'pref_name' AS {MAPPING_TYPE}
            FROM molecule_dictionary
        """
        df = pd.read_sql(query, conn)
        # eliminate anything without a pref_name, as will be too big otherwise
        df = df.dropna(subset=[DEFAULT_LABEL])

        df.drop_duplicates(inplace=True)

        return df


class ChemblParquetOntologyParser(OntologyParser):
    """Input is a parquet file containing an extracted sqllite dump from Chembl.

    .. note::
       See :class:`kazu.ontology_preprocessing.downloads.ChemblParquetOntologyDownloader` for how the extraction is performed.
    """

    def find_kb(self, string: str) -> str:
        return "CHEMBL"

    def parse_to_dataframe(self) -> pd.DataFrame:
        df = pd.read_parquet(self.in_path)
        return df


class CLOOntologyParser(RDFGraphParser):
    """Input is a CLO Owl file https://www.ebi.ac.uk/ols/ontologies/clo."""

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        ontology_downloader: Optional[OntologyDownloader] = None,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            uri_regex=re.compile("^http://purl.obolibrary.org/obo/CLO_[0-9]+$"),
            synonym_predicates=(
                rdflib.URIRef("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"),
            ),
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            curations_path=curations_path,
            global_actions=global_actions,
            autocurator=autocurator,
            ontology_downloader=ontology_downloader,
        )

    def find_kb(self, string: str) -> str:
        return "CLO"


class CellosaurusOntologyParser(OntologyParser):
    """Input is an obo file from cellosaurus, e.g.
    https://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo."""

    cell_line_re = re.compile("cell line", re.IGNORECASE)

    def find_kb(self, string: str) -> str:
        return "CELLOSAURUS"

    def score_and_group_ids(
        self,
        ids_and_source: IdsAndSource,
        is_symbolic: bool,
    ) -> tuple[AssociatedIdSets, EquivalentIdAggregationStrategy]:
        """Treat all synonyms as seperate cell lines.

        :param ids_and_source:
        :param is_symbolic:
        :return:
        """

        return (
            frozenset(
                EquivalentIdSet(
                    ids_and_source=frozenset((single_id_and_source,)),
                )
                for single_id_and_source in ids_and_source
            ),
            EquivalentIdAggregationStrategy.CUSTOM,
        )

    def _remove_cell_line_text(self, text: str) -> str:
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
    """Input is an unzipped directory to a Meddra release (Note, requires licence).

    This should contain the files 'mdhier.asc' and 'llt.asc'.
    """

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        ontology_downloader: Optional[OntologyDownloader] = None,
        exclude_socs: Iterable[str] = (
            "Surgical and medical procedures",
            "Social circumstances",
            "Investigations",
        ),
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            curations_path=curations_path,
            global_actions=global_actions,
            name=name,
            autocurator=autocurator,
            ontology_downloader=ontology_downloader,
        )

        self.exclude_socs = exclude_socs

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
        hier_df = hier_df[~hier_df["soc_name"].isin(self.exclude_socs)]

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
        soc_names = []
        soc_codes = []

        for i, row in hier_df.iterrows():
            idx = row["pt_code"]
            pt_name = row["pt_name"]
            soc_name = row["soc_name"]
            soc_code = row["soc_code"]
            llts = llt_df[llt_df["pt_code"] == idx]
            ids.append(idx)
            default_labels.append(pt_name)
            all_syns.append(pt_name)
            soc_names.append(soc_name)
            soc_codes.append(soc_code)
            mapping_type.append("meddra_link")
            for j, llt_row in llts.iterrows():
                ids.append(idx)
                default_labels.append(pt_name)
                soc_names.append(soc_name)
                soc_codes.append(soc_code)
                all_syns.append(llt_row["llt_name"])
                mapping_type.append("meddra_link")

        for i, row in (
            hier_df[["hlt_code", "hlt_name", "soc_name", "soc_code"]].drop_duplicates().iterrows()
        ):
            ids.append(row["hlt_code"])
            default_labels.append(row["hlt_name"])
            soc_names.append(row["soc_name"])
            soc_codes.append(row["soc_code"])
            all_syns.append(row["hlt_name"])
            mapping_type.append("meddra_link")
        for i, row in (
            hier_df[["hlgt_code", "hlgt_name", "soc_name", "soc_code"]].drop_duplicates().iterrows()
        ):
            ids.append(row["hlgt_code"])
            default_labels.append(row["hlgt_name"])
            soc_names.append(row["soc_name"])
            soc_codes.append(row["soc_code"])
            all_syns.append(row["hlgt_name"])
            mapping_type.append("meddra_link")
        df = pd.DataFrame.from_dict(
            {
                IDX: ids,
                DEFAULT_LABEL: default_labels,
                SYN: all_syns,
                MAPPING_TYPE: mapping_type,
                "soc_name": soc_names,
                "soc_code": soc_codes,
            }
        )
        return df


class CLOntologyParser(RDFGraphParser):
    """Input should be an CL owl file e.g.
    https://www.ebi.ac.uk/ols/ontologies/cl."""

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.7,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        include_entity_patterns: EntityFilteringPatterns = None,
        exclude_entity_patterns: EntityFilteringPatterns = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        ontology_downloader: Optional[OntologyDownloader] = None,
    ):

        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            uri_regex=re.compile("^http://purl.obolibrary.org/obo/CL_[0-9]+$"),
            synonym_predicates=(
                rdflib.URIRef("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"),
            ),
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            include_entity_patterns=include_entity_patterns,
            exclude_entity_patterns=exclude_entity_patterns,
            curations_path=curations_path,
            global_actions=global_actions,
            autocurator=autocurator,
            ontology_downloader=ontology_downloader,
        )

    def find_kb(self, string: str) -> str:
        return "CL"


class HGNCGeneFamilyParser(OntologyParser):
    """Parse HGNC data and extract only Gene Families as entities.

    Input is a json from HGNC. For example,
    http://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/json/hgnc_complete_set.json.
    """

    syn_column_keys = {"Family alias", "Common root gene symbol"}

    def find_kb(self, string: str) -> str:
        return "HGNC_GENE_FAMILY"

    def parse_to_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.in_path, sep="\t")
        data = []
        for family_id, row in (
            df.groupby(by="Family ID").agg(lambda col_series: set(col_series.dropna())).iterrows()
        ):
            # in theory, there should only be one family name per ID
            assert len(row["Family name"]) == 1
            default_label = next(iter(row["Family name"]))
            data.append(
                {
                    SYN: default_label,
                    MAPPING_TYPE: "Family name",
                    DEFAULT_LABEL: default_label,
                    IDX: family_id,
                }
            )
            data.extend(
                {
                    SYN: syn,
                    MAPPING_TYPE: key,
                    DEFAULT_LABEL: default_label,
                    IDX: family_id,
                }
                for key in self.syn_column_keys
                for syn in row[key]
            )
        return pd.DataFrame.from_records(data)


class TabularOntologyParser(OntologyParser):
    """For already tabulated data.

    This expects ``in_path`` to be the path to a file that can be loaded
    by :func:`pandas.read_csv` (e.g. a ``.csv`` or ``.tsv`` file),
    and the result be in the format that is produced by
    :meth:`~.OntologyParser.parse_to_dataframe` - see the docs of that
    method for more details on the format of this dataframe.

    Note that this class's ``__init__`` method takes a ``**kwargs`` parameter
    which is passed through to :func:`pandas.read_csv` , which gives you
    a notable degree of flexibility on how exactly the input file is
    converted into this dataframe. Although if this becomes complex to
    pass through in the ``**kwargs``, it may be worth considering
    :ref:`writing-a-custom-parser`.
    """

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.7,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        ontology_downloader: Optional[OntologyDownloader] = None,
        **kwargs: Any,
    ):
        """

        :param in_path:
        :param entity_class:
        :param name:
        :param string_scorer:
        :param synonym_merge_threshold:
        :param data_origin:
        :param synonym_generator:
        :param autocurator:
        :param curations_path:
        :param global_actions:
        :param ontology_downloader:
        :param kwargs: passed to pandas.read_csv
        """
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            curations_path=curations_path,
            global_actions=global_actions,
            autocurator=autocurator,
            ontology_downloader=ontology_downloader,
        )
        self._raw_dataframe: pd.DataFrame = pd.read_csv(self.in_path, **kwargs)

    def parse_to_dataframe(self) -> pd.DataFrame:
        """Assume input file is already in correct format.

        Inherit and override this method if different behaviour is required.

        :return:
        """
        return self._raw_dataframe

    def find_kb(self, string: str) -> str:
        return self.name


class ATCDrugClassificationParser(TabularOntologyParser):
    """Parser for the ATC Drug classification dataset.

    This requires a licence from WHO, available at
    https://www.who.int/tools/atc-ddd-toolkit/atc-classification
    .
    """

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        ontology_downloader: Optional[OntologyDownloader] = None,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            curations_path=curations_path,
            global_actions=global_actions,
            sep="     ",
            header=None,
            names=["code", "level_and_description"],
            # Because the c engine can't handle multi-char sep
            # removing this results in the same behaviour, but
            # pandas logs a warning.
            engine="python",
            autocurator=autocurator,
            ontology_downloader=ontology_downloader,
        )

    levels_to_ignore = {"1", "2", "3"}

    def parse_to_dataframe(self) -> pd.DataFrame:
        # the name of this function changed from applymap to just map in pandas 2.1.0
        # and the old name deprecated. A change in pandas-stubs broke mypy for us. Given that
        # we're otherwise compatible with pandas >=1.0.0, it seemed worth preserving compatibility
        # for this one ugly block of code.
        _pandas_map_function_name = (
            "map"
            if packaging.version.parse(pd.__version__) >= packaging.version.parse("2.1.0rc0")
            else "applymap"
        )
        df: pd.DataFrame = getattr(self._raw_dataframe, _pandas_map_function_name)(str.strip)
        res_df = pd.DataFrame()
        # for some reason, the level and description codes are merged, so we need to fix this here
        res_df[[MAPPING_TYPE, DEFAULT_LABEL]] = df.apply(
            lambda row: [row["level_and_description"][0], row["level_and_description"][1:]],
            axis=1,
            result_type="expand",
        )
        res_df[IDX] = df["code"]
        res_df = res_df[~res_df[MAPPING_TYPE].isin(self.levels_to_ignore)]
        res_df[SYN] = res_df[DEFAULT_LABEL]
        return res_df


class StatoParser(RDFGraphParser):
    """Parse stato: input should be an owl file.

    Available at e.g.
    https://www.ebi.ac.uk/ols/ontologies/stato .
    """

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.7,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        include_entity_patterns: EntityFilteringPatterns = None,
        exclude_entity_patterns: EntityFilteringPatterns = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        ontology_downloader: Optional[OntologyDownloader] = None,
    ):

        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            uri_regex=re.compile("^http://purl.obolibrary.org/obo/(OBI|STATO)_[0-9]+$"),
            synonym_predicates=(rdflib.URIRef("http://purl.obolibrary.org/obo/IAO_0000111"),),
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            include_entity_patterns=include_entity_patterns,
            exclude_entity_patterns=exclude_entity_patterns,
            curations_path=curations_path,
            global_actions=global_actions,
            autocurator=autocurator,
            ontology_downloader=ontology_downloader,
        )

    def find_kb(self, string: str) -> str:
        return "OBI" if "OBI" in string else "STATO"


class HPOntologyParser(RDFGraphParser):
    def find_kb(self, string: str) -> str:
        return "HP"
