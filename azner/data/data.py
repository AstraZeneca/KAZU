import uuid
from typing import List, Any, Dict, Optional

from pydantic import BaseModel, Field, validator


class Mapping(BaseModel):
    source: str  # the knowledgebase name
    idx: str  # the identifier within the KB
    mapping_type: str  # the type of KB mapping
    metadata: Optional[Dict[Any, Any]] = Field(default_factory=dict, hash=False)  # generic metadata


class EntityMetadata(BaseModel):
    mappings: Optional[List[Mapping]]  # KB mappings
    metadata: Optional[Dict[Any, Any]] = Field(default_factory=dict, hash=False)  # generic metadata


class Entity(BaseModel):
    """
    Generic data class representing a unique entity in a string
    """

    namespace: str  # namespace of BaseStep that produced this instance
    match: str  # exact text representation
    entity_class: str  # entity class
    start: int  # start offset
    end: int  # end offset
    hash_val: Optional[str] = None  # not required. calculated based on above fields
    metadata: Optional[EntityMetadata] = Field(
        default_factory=EntityMetadata, hash=False
    )  # generic metadata

    @validator("hash_val", always=True)
    def populate_hash(cls, v, values):
        return hash(
            (
                values.get("namespace"),
                values.get("match"),
                values.get("entity_class"),
                values.get("start"),
                values.get("end"),
            )
        )

    def exact_overlap(self, other, also_same_class: bool = False):
        """
        compare one entity to other

        :param other: query Entity
        :param also_same_class: if True, will return true if entity_class matches
        """
        if (
            also_same_class
            and self.entity_class == other.entity_class
            and all([self.start == other.start, self.end == other.end])
        ):
            return True
        else:
            return all([self.start == other.start, self.end == other.end])

    def __hash__(self):
        return self.hash_val

    def __eq__(self, other):
        return other.hash == self.hash_val

    def overlapped(self, other) -> bool:
        """
        Test for distinct offsets
        :param other: query Entity
        :return: True if the offsets are completely distinct, False otherwise
        """
        return (self.end <= other.start) or (self.start >= other.end)

    def __len__(self) -> int:
        """
        Span length
        :return: number of characters enclosed by span
        """
        return self.end - self.start

    def __repr__(self) -> str:
        """
        Describe the tag
        :return: tag match description
        """
        return f"{self.namespace}:{self.match}:{self.entity_class}:{self.metadata}:{self.match}:{self.start}:{self.end}"

    def as_brat(self):
        """
        self as the third party biomedical nlp Brat format, (see docs on Brat)
        """
        return f"{self.hash_val}\t{self.entity_class}\t{self.start}\t{self.end}\t{self.match}\n"

    def add_mapping(self, mapping: Mapping):
        if self.metadata.mappings is None:
            self.metadata.mappings = [mapping]
        else:
            self.metadata.mappings.append(mapping)


class Section(BaseModel):
    text: str  # the text to be processed
    name: str  # the name of the section (e.g. abstract, body, header, footer etc)
    hash_val: Optional[str] = None  # not required. calculated based on above fields
    metadata: Optional[Dict[Any, Any]] = Field(default_factory=dict, hash=False)  # generic metadata
    entities: List[Entity] = Field(
        default_factory=list, hash=False
    )  # entities detected in this section

    def __str__(self):
        return f"name: {self.name}, text: {self.text[:100]}"

    def __hash__(self):
        return self.hash_val

    @validator("hash_val", always=True)
    def populate_hash(cls, v, values):
        return hash((values.get("text"), values.get("name")))


class Document(BaseModel):
    idx: str  # a document identifier. Note, if you only want to process text strings, use SimpleDocument
    hash_val: Optional[str] = None  # calculated automatically based on above fields
    sections: List[Section]  # sections comprising this document
    metadata: Optional[Dict[Any, Any]] = Field(default_factory=dict, hash=False)  # generic metadata

    def __str__(self):
        return f"idx: {self.idx}"

    def __hash__(self):
        return self.hash_val

    @validator("hash_val", always=True)
    def populate_hash(cls, v, values):
        return hash(values.get("idx"))

    def get_entities(self) -> List[Entity]:
        """
        get all entities in this document
        :return:
        """
        entities = []
        for section in self.sections:
            entities.extend(section.entities)
        return entities


class SimpleDocument(Document):
    """
    a simplified Document. Use this if you want to process a str of text.
    idx is calculated by uuid.uuid4().hex
    the instance will have a single section, composed of the argument to the constructor
    """

    def __init__(self, text: str):
        idx = uuid.uuid4().hex
        sections = [Section(text=text, name="na")]
        super().__init__(idx=idx, sections=sections)

    @validator("hash_val", always=True)
    def populate_hash(cls, v, values):
        return hash(values.get("idx"))
