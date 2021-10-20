from typing import List, Dict, Tuple

from transformers import AutoTokenizer, BatchEncoding

from azner.data.data import Document, Entity


def documents_to_document_section_text_map(docs: List[Document]) -> Dict[str, str]:
    """
    convert documents into a dict of <dochash + sectionhash>: text
    :param docs:
    :return:
    """
    return {
        (doc.hash_val + section.hash_val): section.text for doc in docs for section in doc.sections
    }


def documents_to_entity_list(docs: List[Document]) -> List[Entity]:
    """
    takes a list of documents and returns all entities found within them
    :param docs:
    :return:
    """
    entities = []
    for doc in docs:
        for section in doc.sections:
            entities.extend(section.entities)
    return entities


def filter_entities_with_kb_mappings(entities: List[Entity]) -> List[Entity]:
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
