from typing import List, Dict, Tuple

from transformers import AutoTokenizer, BatchEncoding

from azner.data.data import Document


def documents_to_document_section_map(docs: List[Document]):
    return {(doc.hash_val + section.hash_val): section for doc in docs for section in doc.sections}


def documents_to_document_section_text_map(docs: List[Document]) -> Dict[str, str]:
    return {
        (doc.hash_val + section.hash_val): section.text for doc in docs for section in doc.sections
    }


def documents_to_document_section_text_tuples(docs: List[Document]):
    return {
        (doc.hash_val + section.hash_val): section.text for doc in docs for section in doc.sections
    }


def documents_to_document_section_batch_encodings_map(
    docs: List[Document], tokenizer: AutoTokenizer
) -> Tuple[BatchEncoding, List[str]]:
    doc_section_to_text_map = documents_to_document_section_text_map(docs)
    batch_encodings = tokenizer(list(doc_section_to_text_map.values()))
    return batch_encodings, list(doc_section_to_text_map.keys())
