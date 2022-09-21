import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Set, Tuple, DefaultDict, Dict, List

import requests

from kazu.data.data import Mapping
from kazu.steps.linking.post_processing.mapping_strategies.strategies import MappingFactory
from kazu.utils.utils import get_cache_dir

logger = logging.getLogger(__name__)

XRefDB = Dict[str, Dict[str, List[Tuple[str, str]]]]


def _serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    return obj


SourceOntology = str
SourceIdx = str

TargetSourceAndIdx = Tuple[str, str]
ToSourceAndIDXMap = Dict[SourceIdx, List[TargetSourceAndIdx]]
XrefDatabase = Dict[SourceOntology, ToSourceAndIDXMap]


class CrossReferenceManager(ABC):
    def __init__(self, source_to_parser_metadata_lookup: Dict[str, str], path: Path):
        """
        :param source_to_parser_metadata_lookup: when producing cross-referenced instances of Mapping, we need a
            reference in the MetadataDatabase to the target ontology, in order to look up the default label info etc.
            This lookup dict tells the cross reference manager what underlying parser it should use for a given source,
            since different parsers may hold sub sets or supersets of ids of each other. For example, a MedDRA hit
            might map to specific MONDO id. Since MONDO ids are held in both OpenTargetsDiseaseOntologyParser and
            MondoOntologyParser, we need to specify which one we want to use to generate the mapping
        :param path: path to cross ref mapping resources required by this manager
        """
        self.source_to_parser_metadata_lookup = source_to_parser_metadata_lookup
        self.load_or_build_cache(path)

    @abstractmethod
    def build_xref_cache(self, path: Path) -> XrefDatabase:
        """
        build a XrefDatabase suitable for caching

        :param path:
        :param cache_path:
        :return:
        """
        raise NotImplementedError()

    def load_or_build_cache(self, path: Path, force_rebuild_cache: bool = False):
        """
        build the index, or if a cached version is available, load it instead
        """
        cache_dir = get_cache_dir(
            path,
            prefix=f"{self.__class__.__name__}",
            create_if_not_exist=False,
        )
        if force_rebuild_cache:
            logger.info("forcing a rebuild of the cache")
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            xref_db = self.build_xref_cache(path)
            self.save(cache_dir, xref_db)
            self.load(cache_dir)
        elif cache_dir.exists():
            logger.info(f"loading cached file from {cache_dir}")
            self.load(cache_dir)
        else:
            logger.info("No cache file found. Building a new one")
            xref_db = self.build_xref_cache(path)
            self.save(cache_dir, xref_db)
            self.load(cache_dir)

    def save(self, cache_path: Path, xref_db: XrefDatabase) -> Path:
        """
        save to disk. Makes a directory at the path location with all the index assets

        :param directory: a dir to save the index.
        :param overwrite: should the directory be deleted before attempting to save? (CAREFUL!)
        :return: a Path to where the index was saved
        """
        if cache_path.exists():
            raise RuntimeError(f"{cache_path} already exists")

        os.makedirs(cache_path, exist_ok=False)
        with open(cache_path.joinpath("xref_db.json"), "w") as f:
            json.dump(xref_db, f, default=_serialize_sets)
        return cache_path

    def load(self, cache_path: Path):
        """
        load from disk

        :param cache_path: the path to the cached files. Normally created via .save
        :return:
        """
        with open(cache_path.joinpath("xref_db.json"), "r") as f:
            self.xref_db = json.load(f)

    def create_xref_mappings(self, mapping: Mapping) -> Iterable[Mapping]:
        xref_lookup: ToSourceAndIDXMap = self.xref_db.get(mapping.source, {})
        for (target_source, target_idx) in xref_lookup.get(mapping.idx, []):
            metadata_parser_name = self.source_to_parser_metadata_lookup.get(target_source)
            if metadata_parser_name is None:
                logger.warning(
                    f"source_to_parser_metadata_lookup not configured for target source: {metadata_parser_name}"
                )
            else:
                try:
                    xref_mapping = MappingFactory.create_mapping(
                        parser_name=metadata_parser_name,
                        xref_source_parser_name=mapping.parser_name,
                        source=target_source,
                        idx=target_idx,
                        mapping_strategy=self.__class__.__name__,
                        disambiguation_strategy=mapping.disambiguation_strategy,
                        confidence=mapping.confidence,
                        additional_metadata=mapping.metadata,
                    )
                    yield xref_mapping
                except KeyError:
                    logger.debug(
                        "failed to create xref mapping for %s->%s:%s->%s. Metadata not found.",
                        mapping.parser_name,
                        metadata_parser_name,
                        mapping.idx,
                        target_idx,
                    )


class OxoCrossReferenceManager(CrossReferenceManager):
    """
    A CrossReferenceManager that uses the EBI OXO service to identify cross-references. Downloads a set of
    cross-references directly from EBI, and caches locally

    """

    def __init__(
        self,
        source_to_parser_metadata_lookup: Dict[str, str],
        path: Path,
        oxo_kazu_name_mapping: Dict[str, str],
        uri_prefixes: Dict[str, str],
        oxo_query: Dict[str, List[str]],
    ):
        """

        :param oxo_kazu_name_mapping: mapping of OXO source names to Kazu names, to covert OXO format to Kazu. If
            not specified, the OXO version will be used
        :param uri_prefixes: mapping of KAZU sources to URI prefixes, to correctly reconstruct ids. If not specified,
            no prefix will be used
        :param oxo_query: mapping of OXO source to target sources that will be used to construct the OXO API request
        """

        self.oxo_query = oxo_query
        self.uri_prefixes = uri_prefixes
        self.oxo_kazu_name_mapping = oxo_kazu_name_mapping
        super().__init__(source_to_parser_metadata_lookup, path)

    def build_xref_cache(self, path: Path) -> XrefDatabase:
        oxo_dump_path = path.joinpath("oxo_dump.json")
        logger.info(f"looking for oxo dump at {oxo_dump_path}")
        if not oxo_dump_path.exists():
            logger.info(f"oxo dump not found. Attempting to download from {self.oxo_url}")
            self.create_oxo_dump(oxo_dump_path)
        logger.info(f"loading from oxo dump at {oxo_dump_path}")
        xref_db = self.parse_oxo_dump(oxo_dump_path)
        return xref_db

    def _convert_oxo_source_string(self, oxo_source: str) -> str:
        return self.oxo_kazu_name_mapping.get(oxo_source, oxo_source)

    def _add_kazu_uri_prefix(self, oxo_idx: str, source: str) -> str:
        if source in {"MONDO", "HP"}:
            return f"{self.uri_prefixes.get(source,'')}{source}_{oxo_idx}"
        else:
            return oxo_idx

    def parse_oxo_dump(self, path: Path) -> XrefDatabase:
        xref_db_default_dict: DefaultDict[
            str, DefaultDict[str, Set[Tuple[str, str]]]
        ] = defaultdict(lambda: defaultdict(set))
        with open(path, "r") as f:
            oxo_dump = json.load(f)
            for oxo_page in oxo_dump:
                for search_result in oxo_page["_embedded"]["searchResults"]:
                    source, idx = search_result["curie"].split(":")
                    source = self._convert_oxo_source_string(source)
                    idx = self._add_kazu_uri_prefix(idx, source)
                    for mapping_response in search_result["mappingResponseList"]:
                        target_source, target_idx = mapping_response["curie"].split(":")
                        target_source = self._convert_oxo_source_string(target_source)
                        target_idx = self._add_kazu_uri_prefix(target_idx, target_source)

                        xref_db_default_dict[source][idx].add((target_source, target_idx))
        xref_db = {}
        for k, v in xref_db_default_dict.items():
            xref_db[k] = {k1: list(v1) for k1, v1 in v.items()}
        return xref_db

    def create_oxo_dump(self, path: Path):
        results = []
        for input_source, mapping_target in self.oxo_query.items():
            data = {
                "ids": [],
                "inputSource": input_source,
                "mappingTarget": mapping_target,
                "distance": "1",
            }
            response = requests.post(f"{self.oxo_url}", headers=self.headers, json=data)
            response_data = response.json()
            link_info = response_data["_links"]
            last = link_info["last"]["href"]
            current = link_info["first"]["href"]

            while current != last:
                response = requests.post(current, headers=self.headers, json=data)
                response_data = response.json()
                link_info = response_data["_links"]
                next = link_info["next"]["href"]
                current = next
                results.append(response_data)
            else:
                response = requests.post(last, headers=self.headers, json=data)
                results.append(response.json())

        with open(path, "w") as f:
            json.dump(results, f)
