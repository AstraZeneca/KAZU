import dataclasses
import json
import tempfile
import uuid
import webbrowser
from dataclasses import dataclass, field
from math import inf
from typing import List, Any, Dict, Optional, Tuple, FrozenSet

import pandas as pd
from numpy import ndarray
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


@dataclass
class CharSpan:
    """A concept similar to a Spacy Span, except is character index based rather than token based"""

    start: int
    end: int

    def is_completely_overlapped(self, other):
        """
        True if other completely overlaps this span
        :param other:
        :return:
        """
        return self.start >= other.start and self.end <= other.end

    def is_partially_overlapped(self, other):
        """
        True if other partially overlaps this span
        :param other:
        :return:
        """
        return (other.start <= self.start <= other.end) or (other.start <= self.end <= other.end)

    def __lt__(self, other):
        return self.start < other.start

    def __gt__(self, other):
        return self.end > other.end

    def __hash__(self):
        return hash((self.start, self.end))


@dataclass
class TokenizedWord:
    """
    A convenient container for a word, which may be split into multiple tokens by e.g. WordPiece tokenisation
    """

    word_labels: List[int] = field(default_factory=list, hash=False)  # label ids of the word
    word_labels_strings: List[str] = field(
        default_factory=list, hash=False
    )  # optional strings associated with labels
    word_confidences: List[float] = field(
        default_factory=list, hash=False
    )  # confidence associated with each label
    word_offsets: List[Tuple[int, int]] = field(
        default_factory=list, hash=False
    )  # original offsets of each token
    bio_labels: List[str] = field(
        default_factory=list, hash=False
    )  # BIO labels separated from class. Populate with parse_labels_to_bio_and_class
    class_labels: List[str] = field(
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


@dataclass
class NerProcessedSection:
    """
    long Sections may need to be split into multiple frames when processing with a transformer. This class is a
    convenient container to reassemble the frames into a coherent object, post transformer
    """

    all_frame_offsets: List[Tuple[int, int]] = field(
        default_factory=list, hash=False
    )  # offsets associated with each token
    all_frame_word_ids: List[int] = field(
        default_factory=list, hash=False
    )  # word ids associated with each token
    all_frame_labels: List[int] = field(default_factory=list, hash=False)  # labels for each token
    all_frame_confidences: List[float] = field(
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


@dataclass
class Mapping:
    default_label: str  # default label from knowledgebase
    source: str  # the knowledgebase name
    idx: str  # the identifier within the KB
    mapping_type: List[str]  # the type of KB mapping
    metadata: Dict[Any, Any] = field(default_factory=dict, hash=False)  # generic metadata


@dataclass
class Entity:
    """
    Generic data class representing a unique entity in a string. Note, since an entity can consist of multiple CharSpan,
    we have to define the semantics of overlapping spans.
    """

    match: str  # exact text representation
    entity_class: str  # entity class
    spans: FrozenSet[CharSpan]  # charspans
    namespace: str  # namespace of BaseStep that produced this instance
    mappings: List[Mapping] = field(default_factory=list, hash=False)
    metadata: Dict[Any, Any] = field(default_factory=dict, hash=False)  # generic metadata
    start: int = field(init=False)
    end: int = field(init=False)

    def calc_starts_and_ends(self) -> Tuple[int, int]:
        earliest_start = inf
        latest_end = 0
        for span in self.spans:
            if span.start < earliest_start:
                earliest_start = span.start
            if span.end > latest_end:
                latest_end = span.end
        if earliest_start is inf:
            raise ValueError("spans are not valid")
        return int(earliest_start), latest_end

    def __post_init__(self):
        self.start, self.end = self.calc_starts_and_ends()

    def is_completely_overlapped(self, other):
        """
        True if all CharSpan instances are completely encompassed by all other CharSpan instances
        :param other:
        :return:
        """
        for charspan in self.spans:
            for other_charspan in other.spans:
                if charspan.is_completely_overlapped(other_charspan):
                    break
            else:
                return False
        return True

    def is_partially_overlapped(self, other):
        """
        True if only one CharSpan instance is defined in both self and other, and they are partially overlapped

        If multiple CharSpan are defined in both self and other, this becomes pathological, as while they may overlap
        in the technical sense, they may have distinct semantic meaning. For instance, consider the case where
        we may want to use is_partially_overlapped to select the longest annotation span suggested
        by some NER system.

        case 1:
        text: the patient has metastatic liver cancers
        entity1:  metastatic liver cancer -> [CharSpan(16,39]
        entity2: liver cancers -> [CharSpan(27,40]

        result: is_partially_overlapped -> True (entities are part of same concept)


        case 2: non-contiguous entities

        text: lung and liver cancer
        lung cancer -> [CharSpan(0,4), CharSpan(1521
        liver cancer -> [CharSpan(9,21)]

        result: is_partially_overlapped -> False (entities are distinct)

        :param other:
        :return:
        """
        if len(self.spans) == 1 and len(other.spans) == 1:
            (charspan,) = self.spans
            (other_charspan,) = other.spans
            return charspan.is_partially_overlapped(other_charspan)
        else:
            return False

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
        return f"{self.match}:{self.entity_class}:{self.namespace}:{self.start}:{self.end}"

    def as_brat(self):
        """
        :return: self as the third party biomedical nlp Brat format, (see docs on Brat)
        """
        # TODO: update this to make use of non-contiguous entities
        return f"{hash(self)}\t{self.entity_class}\t{self.start}\t{self.end}\t{self.match}\n"

    def add_mapping(self, mapping: Mapping):
        """
        deprecated
        :param mapping:
        :return:
        """
        self.mappings.append(mapping)

    @classmethod
    def from_spans(cls, spans: List[Tuple[int, int]], text: str, **kwargs):
        """
        create an instance of Entity from a list of character indices. A text string of underlying doc is
        also required to produce a representative match
        :param spans:
        :param text:
        :param kwargs:
        :return:
        """
        char_spans = []
        text_pieces = []
        for start, end in spans:
            text_pieces.append(text[start:end])
            char_spans.append(CharSpan(start=start, end=end))
        return cls(spans=frozenset(char_spans), match=" ".join(text_pieces), **kwargs)

    @classmethod
    def load_contiguous_entity(cls, start: int, end: int, **kwargs) -> "Entity":
        single_span = frozenset([CharSpan(start=start, end=end)])
        return cls(spans=single_span, **kwargs)


@dataclass
class Section:
    text: str  # the text to be processed
    name: str  # the name of the section (e.g. abstract, body, header, footer etc)

    preprocessed_text: Optional[
        str
    ] = None  # not required. a string representing text that has been preprocessed by e.g. abbreviation expansion
    offset_map: Dict[CharSpan, CharSpan] = field(
        default_factory=dict, hash=False, init=False
    )  # not required. if a preprocessed_text is used, this represents mappings of the preprocessed charspans back to the original

    metadata: Optional[Dict[Any, Any]] = field(default_factory=dict, hash=False)  # generic metadata
    entities: List[Entity] = field(
        default_factory=list, hash=False
    )  # entities detected in this section

    def __str__(self):
        return f"name: {self.name}, text: {self.get_text()[:100]}"

    def get_text(self) -> str:
        """
        rather than accessing text or preprocessed_text directly, this method provides a convenient wrapper to get
        preprocessed_text if available, or text if not.
        :return:
        """
        if self.preprocessed_text is None:
            return self.text
        else:
            return self.preprocessed_text

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
            if len(entity.mappings) > 0:
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

    def entities_as_dataframe(self) -> Optional[pd.DataFrame]:
        """
        convert entities into a pandas dataframe. Useful for building annotation sets
        non-contiguous entities currently not supported
        :return:
        """
        data = []
        for ent in self.entities:
            if len(ent.mappings) > 0:
                mapping_id = ent.mappings[0].idx
                mapping_label = ent.mappings[0].metadata["default_label"]
                mapping_conf = ent.mappings[0].metadata[LINK_CONFIDENCE]
                metadata = ent.mappings[0].metadata
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
            df: pd.DataFrame = pd.DataFrame.from_records(data)
            df.columns = [
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
            return df
        else:
            return None


class DocumentEncoder(json.JSONEncoder):
    """
    Since the Document model can't be directly serialised to JSON, we need a custom encoder/decoder
    """

    def default(self, obj):
        if isinstance(obj, Section):
            as_dict = obj.__dict__
            # needed as CharSpan keys in offset map are not json serialisable
            if as_dict.get("offset_map"):
                as_dict["offset_map"] = list(as_dict["offset_map"].items())
            return as_dict
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif isinstance(obj, ndarray):
            return obj.tolist()
        elif dataclasses.is_dataclass(obj):
            return obj.__dict__
        else:
            return json.JSONEncoder.default(self, obj)


@dataclass
class Document:
    idx: str  # a document identifier
    sections: List[Section] = field(
        default_factory=list, hash=False
    )  # sections comprising this document
    metadata: Dict[Any, Any] = field(default_factory=dict, hash=False)  # generic metadata

    def __str__(self):
        return f"idx: {self.idx}"

    def get_entities(self) -> List[Entity]:
        """
        get all entities in this document
        :return:
        """
        entities = []
        for section in self.sections:
            entities.extend(section.entities)
        return entities

    def json(self, **kwargs):
        """
        custom encoder needed to handle serialisation issues with our data model
        :param kwargs: additional kwargs passed to json.dumps
        :return:
        """
        return json.dumps(self, cls=DocumentEncoder, **kwargs)

    @classmethod
    def create_simple_document(cls, text: str) -> "Document":
        """
        create an instance of document from a text string. The idx field will be generated from uuid.uuid4().hex
        :param text:
        :return:
        """
        idx = uuid.uuid4().hex
        sections = [Section(text=text, name="na")]
        return cls(idx=idx, sections=sections)
