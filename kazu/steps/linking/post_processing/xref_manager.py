import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from collections.abc import Iterable

import requests
from requests.adapters import HTTPAdapter, Retry

from kazu.utils.caching import kazu_disk_cache
from kazu.data import Mapping, IdsAndSource
from kazu.steps.linking.post_processing.mapping_strategies.strategies import MappingFactory

logger = logging.getLogger(__name__)


SourceOntology = str
SourceIdx = str

TargetSourceAndIdx = tuple[str, str]
ToSourceAndIDXMap = dict[SourceIdx, list[TargetSourceAndIdx]]
XrefDatabase = dict[SourceOntology, ToSourceAndIDXMap]


def request_with_retry(url: str, headers: dict, json_data: dict) -> requests.Response:
    s = requests.Session()

    retries = Retry(
        total=25,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"],
    )

    s.mount("https://", HTTPAdapter(max_retries=retries))

    return s.post(url=url, headers=headers, json=json_data)


class CrossReferenceManager(ABC):

    xref_db: XrefDatabase

    def __init__(self, source_to_parser_metadata_lookup: dict[str, str], path: Path):
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
        self.xref_db = self.build_xref_cache(path)

    @abstractmethod
    def build_xref_cache(self, path: Path) -> XrefDatabase:
        """Build a XrefDatabase suitable for caching.

        :param path:
        :return:
        """
        pass

    def create_xref_mappings(self, mapping: Mapping) -> Iterable[Mapping]:
        """Attempt to create additional xref mappings from a source mapping.

        :param mapping:
        :return:
        """
        xref_lookup = self.xref_db.get(mapping.source, {})
        for (target_source, target_idx) in xref_lookup.get(mapping.idx, []):
            metadata_parser_name = self.source_to_parser_metadata_lookup.get(target_source)
            if metadata_parser_name is None:
                logger.warning(
                    f"source_to_parser_metadata_lookup not configured for target source: {metadata_parser_name}"
                )
                continue

            try:
                xref_mapping = MappingFactory.create_mapping(
                    parser_name=metadata_parser_name,
                    xref_source_parser_name=mapping.parser_name,
                    source=target_source,
                    idx=target_idx,
                    string_match_strategy=self.__class__.__name__,
                    disambiguation_strategy=mapping.disambiguation_strategy,
                    disambiguation_confidence=mapping.disambiguation_confidence,
                    string_match_confidence=mapping.string_match_confidence,
                    additional_metadata=mapping.metadata,
                )
                yield xref_mapping
            except KeyError:
                # note, we've set this to debug, as xref mapping sources (e.g. OXO) generally aren't version aware
                # i.e. this will probably fire a lot, and overload the logs if set anything higher than debug.
                # a custom log config at some point would probably help...
                logger.debug(
                    "failed to create xref mapping for %s->%s:%s->%s. Metadata not found.",
                    mapping.parser_name,
                    metadata_parser_name,
                    mapping.idx,
                    target_idx,
                )


class OxoCrossReferenceManager(CrossReferenceManager):
    """A CrossReferenceManager that uses the EBI OXO service to identify cross-
    references.

    Downloads a set of cross-references directly from EBI, and caches locally
    """

    oxo_url = "https://www.ebi.ac.uk/spot/oxo/api/search"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    def __init__(
        self,
        source_to_parser_metadata_lookup: dict[str, str],
        path: Path,
        oxo_kazu_name_mapping: dict[str, str],
        uri_prefixes: dict[str, str],
        oxo_query: dict[str, list[str]],
    ):
        """

        :param source_to_parser_metadata_lookup:
        :param path:
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

    @kazu_disk_cache.memoize(ignore={0})
    def build_xref_cache(self, path: Path) -> XrefDatabase:
        oxo_dump_path = path.joinpath("oxo_dump.json")
        logger.info(f"looking for oxo dump at {oxo_dump_path}")
        if oxo_dump_path.exists():
            with open(oxo_dump_path, "r") as f:
                oxo_dump = json.load(f)
        else:
            logger.info(f"oxo dump not found. Attempting to download from {self.oxo_url}")
            oxo_dump = self.create_oxo_dump(oxo_dump_path)
        logger.info(f"loading from oxo dump at {oxo_dump_path}")
        xref_db = self.parse_oxo_dump(oxo_dump)
        return xref_db

    def _split_and_convert_curie(self, curie: str) -> tuple[str, str]:
        """
        :param curie: a Compact uniform resource identifier, or CURIE. This is essentially a
            prefix followed by a colon (:) followed by a local ID
        :return: A source and idx, converted according to the class's oxo_kazu_name_mapping and uri_prefixes
        """
        oxo_source, oxo_idx = curie.split(":")
        converted_source = self.oxo_kazu_name_mapping.get(oxo_source, oxo_source)
        converted_idx = f"{self.uri_prefixes.get(converted_source,'')}{oxo_idx}"

        return converted_source, converted_idx

    def parse_oxo_dump(self, oxo_dump: list[dict]) -> XrefDatabase:
        xref_db_default_dict: defaultdict[str, defaultdict[str, IdsAndSource]] = defaultdict(
            lambda: defaultdict(set)
        )

        for oxo_page in oxo_dump:
            for search_result in oxo_page["_embedded"]["searchResults"]:
                source, idx = self._split_and_convert_curie(search_result["curie"])
                xref_db_default_dict[source][idx].update(
                    self._split_and_convert_curie(mapping_response["curie"])
                    for mapping_response in search_result["mappingResponseList"]
                )
        xref_db = {}
        for source, id_to_xref in xref_db_default_dict.items():
            xref_db[source] = {idx: list(xref) for idx, xref in id_to_xref.items()}
        return xref_db

    def create_oxo_dump(self, path: Path) -> list[dict]:
        results = []
        for input_source, mapping_target in self.oxo_query.items():
            data = {
                "ids": [],
                "inputSource": input_source,
                "mappingTarget": mapping_target,
                "distance": "1",
            }
            response = request_with_retry(
                url=f"{self.oxo_url}", headers=self.headers, json_data=data
            )
            response_data = response.json()
            link_info = response_data["_links"]
            last = link_info["last"]["href"]
            current = link_info["first"]["href"]

            while current != last:
                response = request_with_retry(url=current, headers=self.headers, json_data=data)
                response_data = response.json()
                link_info = response_data["_links"]
                next = link_info["next"]["href"]
                current = next
                results.append(response_data)
            else:
                response = request_with_retry(url=last, headers=self.headers, json_data=data)
                results.append(response.json())

        with open(path, "w") as f:
            json.dump(results, f)

        return results
