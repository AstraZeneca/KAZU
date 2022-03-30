import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Union, Iterable

from kazu.modelling.ontology_preprocessing.base import DEFAULT_LABEL, MAPPING_TYPE, MetadataDatabase
from transformers import AutoTokenizer, BatchEncoding
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from kazu.data.data import (
    Document,
    Entity,
    Section,
    Hit,
    Mapping,
    LINK_CONFIDENCE,
    LinkRanks,
    AMBIGUOUS_IDX,
)

logger = logging.getLogger(__name__)


class ParseSourceFromId:
    def __init__(self, regex_to_source: Dict[str, Dict[str, str]]):
        self.regex_to_source = defaultdict(dict)
        for source, re_dict in regex_to_source.items():
            for k, v in re_dict.items():
                self.regex_to_source[source][re.compile(k)] = v

    def __call__(self, parser_name: str, idx: str):
        for regex, target_source in self.regex_to_source.get(parser_name, {}).items():
            if re.search(regex, idx):
                return target_source
        else:
            return parser_name


class HitResolver:
    """
    if a hit isn't ambiguous, convert to a mapping
    """

    def __init__(self, id_parser: ParseSourceFromId):
        self.id_parser = id_parser
        self.metadata_db = MetadataDatabase()

    def __call__(self, hit: Hit) -> Iterable[Mapping]:

        if len(hit.syn_data) == 1:
            syn_data = next(iter(hit.syn_data))
            for idx in syn_data.ids:
                metadata = self.metadata_db.get_by_idx(hit.source, idx)
                default_label = metadata.pop(DEFAULT_LABEL, "na")
                mapping_type = frozenset([str(x) for x in metadata.pop(MAPPING_TYPE, ["na"])])
                metadata[LINK_CONFIDENCE] = hit.confidence
                yield Mapping(
                    default_label=str(default_label),
                    source=self.id_parser(hit.source, idx),
                    idx=idx,
                    mapping_type=mapping_type,
                    metadata=metadata,
                )


def find_document_from_entity(docs: List[Document], entity: Entity) -> Document:
    """
    for a given entity and a list of docs, find the doc the entity belongs to

    :param list_map:
    :param entity:
    :return:
    """
    for doc in docs:
        if entity in doc.get_entities():
            return doc
    raise RuntimeError(f"Error! Entity {entity}is not attached to a document")


def documents_to_document_section_text_map(docs: List[Document]) -> Dict[Tuple[int, int], str]:
    """
    convert documents into a dict of <dochash + sectionhash>: text

    :param docs:
    :return:
    """
    return {
        (
            hash(doc),
            hash(section),
        ): section.get_text()
        for doc in docs
        for section in doc.sections
    }


def documents_to_id_section_map(docs: List[Document]) -> Dict[int, Section]:
    """
    return a map of documents, indexed by order of sections

    :param docs:
    :return:
    """
    result = {}
    i = 0
    for doc in docs:
        for section in doc.sections:
            result[i] = section
            i += 1
    return result


def filter_entities_with_ontology_mappings(entities: List[Entity]) -> List[Entity]:
    """
    finds entities that have no kb mappings

    :param entities:
    :return:
    """
    return [x for x in entities if len(x.mappings) == 0]


def documents_to_document_section_batch_encodings_map(
    docs: List[Document], tokenizer: AutoTokenizer, stride: int = 128, max_length: int = 512
) -> Tuple[BatchEncoding, Dict[int, Section]]:
    """
    convert documents into a BatchEncoding. Also returns a list of <int + section> for the resulting encoding

    :param docs:
    :return:
    """
    id_section_map = documents_to_id_section_map(docs)
    strings = [section.get_text() for section in id_section_map.values()]

    batch_encodings = tokenizer(
        strings,
        stride=stride,
        max_length=max_length,
        truncation=TruncationStrategy.LONGEST_FIRST,
        return_overflowing_tokens=True,
        padding=PaddingStrategy.MAX_LENGTH,
    )
    return batch_encodings, id_section_map


def get_match_entity_class_hash(ent: Entity) -> int:
    return hash(
        (
            ent.match,
            ent.entity_class,
        )
    )


PathLike = Union[str, Path]

SinglePathLikeOrIterable = Union[PathLike, Iterable[PathLike]]


def as_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)


def get_cache_dir(path: PathLike, prefix: str = "", create_if_not_exist: bool = True) -> Path:
    path = as_path(path)
    new_path = path.with_name(f"cached_{prefix}_{path.name}")
    if create_if_not_exist:
        if new_path.exists():
            logger.info(f"{new_path} already exists. Will not make it")
        else:
            new_path.mkdir()

    return new_path


def get_cache_path(path: PathLike, cache_id: str) -> Path:
    path = as_path(path)
    original_filename = path.name
    cache_dir = get_cache_dir(path, create_if_not_exist=False)
    new_path = cache_dir.joinpath(f"cached_{cache_id}_{original_filename}")
    return new_path


class EntityClassFilter:
    """
    A condition that returns True if a document has any entities that match the class of the required_entity_classes
    """

    def __init__(self, required_entity_classes: Iterable[str]):
        """

        :param required_entity_classes: list of str, specifying entity classes to assess
        """
        self.required_entities = set(required_entity_classes)

    def __call__(self, document: Document) -> bool:
        return any(
            (entity.entity_class in self.required_entities for entity in document.get_entities())
        )
