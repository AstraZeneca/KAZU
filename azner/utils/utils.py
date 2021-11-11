import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple

from azner.data.data import Document, Entity
from transformers import AutoTokenizer, BatchEncoding

logger = logging.getLogger(__name__)


def documents_to_document_section_text_map(docs: List[Document]) -> Dict[str, str]:
    """
    convert documents into a dict of <dochash + sectionhash>: text
    :param docs:
    :return:
    """
    return {
        (doc.hash_val + section.hash_val): section.text for doc in docs for section in doc.sections
    }


def filter_entities_with_ontology_mappings(entities: List[Entity]) -> List[Entity]:
    """
    finds entities that have no kb mappings
    :param entities:
    :return:
    """
    return [x for x in entities if x.metadata.mappings is None]


def documents_to_document_section_batch_encodings_map(
    docs: List[Document], tokenizer: AutoTokenizer
) -> Tuple[BatchEncoding, List[str]]:
    """
    convert documents into a BatchEncoding. Also returns a list of <dochash + sectionhash> for each batch encoding
    :param docs:
    :return:
    """
    doc_section_to_text_map = documents_to_document_section_text_map(docs)
    batch_encodings = tokenizer(list(doc_section_to_text_map.values()))
    return batch_encodings, list(doc_section_to_text_map.keys())


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
