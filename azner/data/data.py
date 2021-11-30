import tempfile
import uuid
import webbrowser
from typing import List, Any, Dict, Optional, Tuple, Union
import pandas as pd
from pydantic import BaseModel, Field, validator
from spacy import displacy

# BIO schema
ENTITY_START_SYMBOL = "B"
ENTITY_INSIDE_SYMBOL = "I"
ENTITY_OUTSIDE_SYMBOL = "O"

# key for Document Processing Failed
PROCESSING_EXCEPTION = "PROCESSING_EXCEPTION"

# key for namespace in metadata
NAMESPACE = "namespace"

# key for linking score
LINK_SCORE = "link_score"
LINK_CONFIDENCE = "link_confidence"


class CharSpan(BaseModel):
    """A concept similar to a Spacy Span, except is offset based rather than token based"""

    start: int
    end: int

    def is_overlapped(self, other):
        return self.start >= other.start and self.end <= other.end

    def __lt__(self, other):
        return self.start < other.start

    def __gt__(self, other):
        return self.end > other.end

    def __hash__(self):
        return hash((self.start, self.end))


class TokenizedWord(BaseModel):
    """
    A convenient container for a word, which may be split into multiple tokens by e.g. WordPiece tokenisation
    """

    word_labels: List[int] = Field(default_factory=list, hash=False)  # label ids of the word
    word_labels_strings: List[str] = Field(
        default_factory=list, hash=False
    )  # optional strings associated with labels
    word_confidences: List[float] = Field(
        default_factory=list, hash=False
    )  # confidence associated with each label
    word_offsets: List[Tuple[int, int]] = Field(
        default_factory=list, hash=False
    )  # original offsets of each token
    bio_labels: Optional[List[str]] = Field(
        default_factory=list, hash=False
    )  # BIO labels separated from class. Populate with parse_labels_to_bio_and_class
    class_labels: Optional[List[str]] = Field(
        default_factory=list, hash=False
    )  # class labels separated from BIO. Populate with parse_labels_to_bio_and_class
    modified_post_inference: bool = (
        False  # has the word been modified poost inference by e.g. BioLabelPreProcessor?
    )

    def parse_labels_to_bio_and_class(self):
        """
        since the BIO schema actually encodes two pieces of info, it's often useful to handle them separately.
        This method parses the BIO labels in self.word_labels_strings and add the fields bio_labels and class labels
        :return:
        """
        self.bio_labels, self.class_labels = map(
            list,
            zip(
                *[
                    x.split("-")  # split for BIO schema - i.e. B-gene, I-gene -> (B,gene), (I,gene)
                    if x is not ENTITY_OUTSIDE_SYMBOL
                    else (
                        ENTITY_OUTSIDE_SYMBOL,
                        None,
                    )
                    for x in self.word_labels_strings
                ]
            ),
        )

    def __repr__(self) -> str:
        return f"MODIFIED:{self.modified_post_inference}. {self.word_labels_strings}\n{self.word_offsets}"


class NerProcessedSection(BaseModel):
    """
    long Sections may need to be split into multiple frames when processing with a transformer. This class is a
    convenient container to reassemble the frames into a coherent object, post transformer
    """

    all_frame_offsets: List[Tuple[int, int]] = Field(
        default_factory=list, hash=False
    )  # offsets associated with each token
    all_frame_word_ids: List[Optional[int]] = Field(
        default_factory=list, hash=False
    )  # word ids associated with each token
    all_frame_labels: List[int] = Field(default_factory=list, hash=False)  # labels for each token
    all_frame_confidences: List[float] = Field(
        default_factory=list, hash=False
    )  # confidence for each token

    def to_tokenized_words(self, id2label: Dict[int, str]) -> List[TokenizedWord]:
        """
        return a List[TokenizedWord]
        :param id2label: Dict mapping labels to strings
        :return:
        """
        prev_word_id = 0
        word = TokenizedWord()
        all_words = []
        for i, word_id in enumerate(self.all_frame_word_ids):
            if word_id is not None:
                if word_id != prev_word_id:
                    # new word
                    all_words.append(word)
                    word = TokenizedWord()
                word.word_labels.append(self.all_frame_labels[i])
                word.word_labels_strings = [id2label[x] for x in word.word_labels]
                word.word_confidences.append(self.all_frame_confidences[i])
                word.word_offsets.append(self.all_frame_offsets[i])
                prev_word_id = word_id

        all_words.append(word)
        return all_words


class Mapping(BaseModel):
    source: str  # the knowledgebase name
    idx: str  # the identifier within the KB
    mapping_type: List[str]  # the type of KB mapping
    metadata: Dict[Any, Any] = Field(default_factory=dict, hash=False)  # generic metadata


class EntityMetadata(BaseModel):
    mappings: List[Mapping] = Field(default_factory=list, hash=False)  # KB mappings
    metadata: Dict[Any, Any] = Field(default_factory=dict, hash=False)  # generic metadata


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
    metadata: EntityMetadata = Field(default_factory=EntityMetadata, hash=False)  # generic metadata

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
        return other.hash_val == self.hash_val

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
        return f"{self.namespace}:{self.match}:{self.entity_class}:{self.metadata}:{self.start}:{self.end}"

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

    preprocessed_text: Optional[
        str
    ] = None  # not required. a string representing text that has been preprocessed by e.g. abbreviation expansion
    offset_map: Optional[
        Union[Dict[CharSpan, CharSpan], Dict[int, List[CharSpan]]]
    ] = None  # not required. if a preprocessed_text is used, this represents mappings of the preprocessed charspans back to the original

    metadata: Optional[Dict[Any, Any]] = Field(default_factory=dict, hash=False)  # generic metadata
    entities: List[Entity] = Field(
        default_factory=list, hash=False
    )  # entities detected in this section

    def __str__(self):
        return f"name: {self.name}, text: {self.get_text()[:100]}"

    def __hash__(self):
        return self.hash_val

    def get_text(self) -> str:
        """
        rather than accessing text or preprocessed_text directly, this method provides a convenient wrapper to get
        preprocessed_text if available, or text if not. We can't use property due to this issue:
        https://github.com/samuelcolvin/pydantic/issues/935
        :return:
        """
        if self.preprocessed_text is None:
            return self.text
        else:
            return self.preprocessed_text

    @validator("hash_val", always=True)
    def populate_hash(cls, v, values):
        return hash((values.get("text"), values.get("name")))

    def render(self):
        ordered_ends = sorted(self.entities, key=lambda x: x.start)
        colors = {
            "gene_linked": "#00f518",
            "gene": "#84fa90",
            "disease_linked": "#ff0000",
            "disease": "#fc7979",
            "drug": "#7d81ff",
            "drug_linked": "#384cff",
            "mutation": "#ff96f5",
        }
        linked = {
            "gene": "gene_linked",
            "disease": "disease_linked",
            "drug": "drug_linked",
        }

        def label_colors(entity: Entity):
            if entity.metadata.mappings is not None and len(entity.metadata.mappings) > 0:
                return linked[entity.entity_class]
            else:
                return entity.entity_class

        ex = [
            {
                "text": self.get_text(),
                "ents": [
                    {"start": x.start, "end": x.end, "label": label_colors(x)} for x in ordered_ends
                ],
                "title": None,
            }
        ]
        html = displacy.render(ex, style="ent", options={"colors": colors}, manual=True)
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as f:
            url = "file://" + f.name
            f.write(html)
        webbrowser.open(url, new=2)

    def as_serialisable(self):
        """
        Since offset map is not json serialisable, we need to convert it something that can be jsonified when using
        the web api. Generally speaking, it's easier to call Document.as_serialisable() which returns a serialisable
        copy of the whole document.
        :return:
        """
        offsets = self.offset_map
        new_section = self.copy(deep=True)
        if offsets is not None and len(offsets) > 0:
            if isinstance(list(offsets.keys())[0], CharSpan):
                serialiable_offsets = {}
                for i, modified_char_span in enumerate(offsets):
                    serialiable_offsets[i] = [modified_char_span, offsets[modified_char_span]]
                new_section.offset_map = serialiable_offsets
            elif isinstance(list(offsets.keys())[0], int):
                # is already serialised
                pass
            else:
                raise RuntimeError("offset map is in indeterminate state")
        return new_section

    def rehydrate(self):
        """
        the inverse of as_serialisable.
        :return:
        """
        new_section = self.copy(deep=True)
        offsets = self.offset_map
        if offsets is not None and len(offsets) > 0:
            if isinstance(list(offsets.keys())[0], int):
                original_offsets = {}
                for i, offsets_list in self.offset_map.items():
                    modified_char_span = CharSpan(
                        start=offsets_list[0]["start"], end=offsets_list[0]["end"]
                    )
                    original_char_span = CharSpan(
                        start=offsets_list[1]["start"], end=offsets_list[1]["end"]
                    )
                    original_offsets[modified_char_span] = original_char_span
                    new_section.offset_map = original_offsets
            elif isinstance(list(offsets.keys())[0], CharSpan):
                # doc is already in normal form
                pass
            else:
                raise RuntimeError("offset map is in indeterminate state")
        return new_section

    def entities_as_dataframe(self) -> Optional[pd.DataFrame]:
        """
        convert entities into a pandas dataframe. Useful for building annotation sets
        :return:
        """
        data = []
        for ent in self.entities:
            if (
                ent.metadata is not None
                and ent.metadata.mappings is not None
                and len(ent.metadata.mappings) > 0
            ):
                mapping_id = ent.metadata.mappings[0].idx
                mapping_label = ent.metadata.mappings[0].metadata["default_label"]
                mapping_conf = ent.metadata.mappings[0].metadata[LINK_CONFIDENCE]
                metadata = ent.metadata.mappings[0].metadata
            else:
                mapping_id = None
                mapping_label = None
                mapping_conf = None
                metadata = None
            data.append(
                (
                    ent.namespace,
                    ent.match,
                    mapping_label,
                    mapping_conf,
                    metadata,
                    mapping_id,
                    ent.entity_class,
                    ent.start,
                    ent.end,
                )
            )
        if len(data) > 0:
            data = pd.DataFrame.from_records(data)
            data.columns = [
                "namespace",
                "match",
                "mapping_label",
                "mapping_conf",
                "metadata",
                "mapping_id",
                "entity_class",
                "start",
                "end",
            ]
            return data
        else:
            return None


class Document(BaseModel):
    idx: str  # a document identifier. Note, if you only want to process text strings, use SimpleDocument
    hash_val: Optional[str] = None  # calculated automatically based on above fields
    sections: List[Section]  # sections comprising this document
    metadata: Dict[Any, Any] = Field(default_factory=dict, hash=False)  # generic metadata

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

    def as_serialisable(self):
        """
        return a copy of the document, in a format that can be json serialised. Intended to be used
        :return:
        """
        new_doc = self.copy()
        for i in range(len(new_doc.sections)):
            new_doc.sections[i] = new_doc.sections[i].as_serialisable()
        return new_doc

    def rehydrate(self):
        """
        the inverse of as_serialisable
        return a copy of the document, in a format that can be json serialised
        :return:
        """
        new_doc = self.copy()
        for i in range(len(new_doc.sections)):
            new_doc.sections[i] = new_doc.sections[i].rehydrate()
        return new_doc


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
