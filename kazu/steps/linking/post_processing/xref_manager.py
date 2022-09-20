import json
import logging
import os
import shutil
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


class CrossReferenceManager:
    oxo_kazu_name_mapping = {"MedDRA": "MEDDRA"}
    uri_prefixes = {
        "MONDO": "http://purl.obolibrary.org/obo/",
        "HP": "http://purl.obolibrary.org/obo/",
    }

    def __init__(self, path: Path, source_to_parser_metadata_lookup: Dict[str, str]):
        """

        :param path: path to oxo mappings dump. Will look for a file called oxo_dump.json in this directory.
            If it doesn't exist, it will try to download it from EBI OXO service
        :param source_to_parser_metadata_lookup: when producing cross-referenced instances of Mapping, we need a
            reference in the MetadataDatabase to the target ontology, in order to look up the default label info etc.
            This lookup dict tells the cross reference manager what underlying parser it should use for a given source,
            since different parsers may hold sub sets or supersets of ids of each other. For example, a MedDRA hit
            might map to specific MONDO id. Since MONDO ids are held in both OpenTargetsDiseaseOntologyParser and
            MondoOntologyParser, we need to specify which one we want to use to generate the mapping
        """
        self.oxo_url = "https://www.ebi.ac.uk/spot/oxo/api/search"
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        self.source_to_parser_metadata_lookup = source_to_parser_metadata_lookup
        self.load_or_build_cache(path)

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
            self.build_xref_cache(path, cache_path=cache_dir)
        elif cache_dir.exists():
            logger.info(f"loading cached file from {cache_dir}")
            self.load(cache_dir)
        else:
            logger.info("No ontology cache file found. Building a new one")
            self.build_xref_cache(path, cache_path=cache_dir)

    def build_xref_cache(self, path: Path, cache_path: Path):
        oxo_dump_path = path.joinpath("oxo_dump.json")
        logger.info(f"looking for oxo dump at {oxo_dump_path}")
        if not oxo_dump_path.exists():
            logger.info(f"oxo dump not found. Attempting to download from {self.oxo_url}")
            self.create_oxo_dump(oxo_dump_path)
        logger.info(f"loading from oxo dump at {oxo_dump_path}")
        xref_db = self.parse_oxo_dump(oxo_dump_path)

        self.save(cache_path, xref_db)
        self.load(cache_path)

    def _map_oxo_source(self, oxo_source: str) -> str:
        return self.oxo_kazu_name_mapping.get(oxo_source, oxo_source)

    def _normalise_idx(self, oxo_idx: str, source: str) -> str:
        if source in {"MONDO", "HP"}:
            return f"{self.uri_prefixes.get(source,'')}{source}_{oxo_idx}"
        else:
            return oxo_idx

    def parse_oxo_dump(
        self, path: Path
    ) -> DefaultDict[str, DefaultDict[str, Set[Tuple[str, str]]]]:
        xref_db: DefaultDict[str, DefaultDict[str, Set[Tuple[str, str]]]] = defaultdict(
            lambda: defaultdict(set)
        )
        with open(path, "r") as f:
            oxo_dump = json.load(f)
            for oxo_page in oxo_dump:
                for search_result in oxo_page["_embedded"]["searchResults"]:
                    source, idx = search_result["curie"].split(":")
                    source = self._map_oxo_source(source)
                    idx = self._normalise_idx(idx, source)
                    for mapping_response in search_result["mappingResponseList"]:
                        target_source, target_idx = mapping_response["curie"].split(":")
                        target_source = self._map_oxo_source(target_source)
                        target_idx = self._normalise_idx(target_idx, target_source)

                        xref_db[source][idx].add((target_source, target_idx))

        return xref_db

    def save(
        self, cache_path: Path, xref_db: DefaultDict[str, DefaultDict[str, Set[Tuple[str, str]]]]
    ) -> Path:
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

    def create_oxo_dump(self, path: Path):
        results = []
        query = {"MONDO": ["MedDRA"], "MedDRA": ["MONDO", "HP"], "HP": ["MedDRA"]}
        for input_source, mapping_target in query.items():
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

    def create_xref_mappings(self, mapping: Mapping, strip_url: bool = True) -> Iterable[Mapping]:
        xref_lookup: Dict[str, Dict] = self.xref_db.get(mapping.source, {})
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
                        strip_url=strip_url,
                    )
                    yield xref_mapping
                except IndexError:
                    logger.warning(
                        f"failed to create xref mapping for "
                        f"{mapping.parser_name}->{metadata_parser_name}:{mapping.idx}->{target_idx}. "
                        f"This is most likely due to a versioning inconsistence between the EBI OXO service"
                        f"and the loaded ontologies"
                    )
