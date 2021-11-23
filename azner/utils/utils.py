import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple

from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from azner.data.data import Document, Entity, Section
from transformers import AutoTokenizer, BatchEncoding

logger = logging.getLogger(__name__)


def find_document_from_entity(docs: List[Document], entity: Entity) -> Document:
    """
    using the result of documents_to_entities_list_map, find the document associated with an Entity
    :param list_map:
    :param entity:
    :return:
    """
    for doc in docs:
        if entity in doc.get_entities():
            return doc
    raise RuntimeError(f"Error! Entity {entity}is not attached to a document")


def documents_to_document_section_text_map(docs: List[Document]) -> Dict[str, str]:
    """
    convert documents into a dict of <dochash + sectionhash>: text
    :param docs:
    :return:
    """
    return {
        (doc.hash_val + section.hash_val): section.get_text()
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
    return [x for x in entities if x.metadata.mappings is None]


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


def get_cache_dir(path: str, create_if_not_exist: bool = True) -> Path:
    path = Path(path)
    original_filename = path.name
    original_dir = path.parent
    new_path = original_dir.joinpath(f"cached_{original_filename}")
    if create_if_not_exist:
        try:
            os.mkdir(new_path)
        except FileExistsError:
            logger.info(f"{new_path} already exists. Will not make it")
    return new_path


def get_cache_path(path_str: str, cache_id: str) -> Path:
    path = Path(path_str)
    original_filename = path.name
    cache_dir = get_cache_dir(path_str, False)
    new_path = cache_dir.joinpath(f"cached_{cache_id}_{original_filename}")
    return new_path
