import dataclasses
import json
import tempfile
import uuid
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import IntEnum, Enum, auto
from itertools import cycle, chain
from math import inf
from typing import List, Any, Dict, Optional, Tuple, FrozenSet

import pandas as pd
from numpy import ndarray, float32, float16
from spacy import displacy


IS_SUBSPAN = "is_subspan"
# ambiguous_synonyms or confused mappings
LINK_UNCERTAINTY = "for_disambiguation"
AMBIGUOUS_IDX = "requires_disambiguation"

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


class LinkRanks(IntEnum):
    # labels for ranking linking hits. NOTE! ordering important, as used for iteration
    HIGH_CONFIDENCE = 0
    MEDIUM_HIGH_CONFIDENCE = 1
    MEDIUM_CONFIDENCE = 2
    AMBIGUOUS = 3
    LOW_CONFIDENCE = 4


class SearchRanks(IntEnum):
    # labels for ranking search hits. NOTE! ordering important, as used for iteration
    EXACT_MATCH = 0
    NEAR_MATCH = 1


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


class EquivalentIdAggregationStrategy(Enum):
    UNAMBIGUOUS = 0
    AMBIGUOUS_WITHIN_SINGLE_KB_SPLIT = 1
    AMBIGUOUS_WITHIN_SINGLE_KB_MERGE = 2
    AMBIGUOUS_ACROSS_MULTIPLE_COMPOSITE_KBS_SPLIT = 3
    AMBIGUOUS_ACROSS_MULTIPLE_COMPOSITE_KBS_MERGE = 4
    AMBIGUOUS_WITHIN_SINGLE_KB_AND_ACROSS_MULTIPLE_COMPOSITE_KBS_SPLIT = 5
    AMBIGUOUS_WITHIN_SINGLE_KB_AND_ACROSS_MULTIPLE_COMPOSITE_KBS_MERGE = 6


@dataclass(frozen=True)
class SynonymData:
    """
    Synonym data is a representation of a set of kb ID's that map to the same synonym and mean the same thing.
    """

    aggregated_by: EquivalentIdAggregationStrategy  # how was this insatnce
    ids: FrozenSet[str] = field(
        default_factory=frozenset, hash=True
    )  # other ID's mapping to this syn, from different KBs
    ids_to_source: Dict[str, str] = field(
        default_factory=dict, hash=False
    )  # needed to lookup the original source of a given id
    mapping_type: FrozenSet[str] = field(default_factory=frozenset, hash=False)

    def __lt__(self, other):
        return tuple(self.ids) < tuple(other.ids)


@dataclass(frozen=True)
class Mapping:
    """
    a mapping is a fully mapped and disambiguated kb concept
    """

    default_label: str  # default label from knowledgebase
    source: str  # the knowledgebase/database/ontology name
    parser_name: str  # the origin of this mapping
    idx: str  # the identifier within the KB
    confidence: LinkRanks
    mapping_type: FrozenSet[str] = field(
        default_factory=frozenset, hash=True
    )  # the type of KB mapping
    metadata: Dict[Any, Any] = field(default_factory=dict, hash=False)  # generic metadata


class SynonymDataSet(frozenset):
    def __new__(cls, data):
        return super(SynonymDataSet, cls).__new__(cls, data)

    def __lt__(self, other):
        return tuple(self) < tuple(other)


@dataclass
class Hit:
    """
    Hit is a precursor to a Mapping, meaning that a match has been detected and linked to a set of SynonymData, but
    is not ready to become a fully fledged mapping yet, as it may require further disambiguation
    """

    string_norm: str
    parser_name: str  # NOTE: this is the parser name, not the kb name. TODO: rename to data_source for consistency
    namespace: str = field(init=False)
    syn_data: FrozenSet[SynonymData]
    confidence: SearchRanks
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)


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
    mappings: List[Mapping] = field(default_factory=list)
    hits: List[Hit] = field(default_factory=list)
    metadata: Dict[Any, Any] = field(default_factory=dict)  # generic metadata
    start: int = field(init=False)
    end: int = field(init=False)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)
        # raise NotImplementedError("entity cannot be compared")
        # return hash((self.entity_class, self.spans))

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
    def from_spans(cls, spans: List[Tuple[int, int]], text: str, join_str: str = "", **kwargs):
        """
        create an instance of Entity from a list of character indices. A text string of underlying doc is
        also required to produce a representative match
        :param spans:
        :param text:
        :param join_str: a string used to join the spans together
        :param kwargs:
        :return:
        """
        char_spans = []
        text_pieces = []
        for start, end in spans:
            text_pieces.append(text[start:end])
            char_spans.append(CharSpan(start=start, end=end))
        return cls(spans=frozenset(char_spans), match=join_str.join(text_pieces), **kwargs)

    @classmethod
    def load_contiguous_entity(cls, start: int, end: int, **kwargs) -> "Entity":
        single_span = frozenset([CharSpan(start=start, end=end)])
        return cls(spans=single_span, **kwargs)


@dataclass(unsafe_hash=True)
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
        colours = cycle(
            ["#00f518", "#84fa90", "#ff0000", "#fc7979", "#7d81ff", "#384cff", "#ff96f5", "#ff96f3"]
        )
        classes = set(chain(*[[x.entity_class, f"{x.entity_class}_linked"] for x in self.entities]))
        colour_map = {k: next(colours) for k in classes}
        linked = {k: f"{k}_linked" for k in classes}

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
        html = displacy.render(ex, style="ent", options={"colors": colour_map}, manual=True)
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
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, ndarray):
            return obj.tolist()
        elif isinstance(obj, float32):
            return obj.item()
        elif isinstance(obj, float16):
            return obj.item()
        elif isinstance(obj, Enum):
            return obj.name
        elif dataclasses.is_dataclass(obj):
            return obj.__dict__
        else:
            return json.JSONEncoder.default(self, obj)


@dataclass(unsafe_hash=True)
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

    def json(self, drop_unmapped_ents: bool = False, drop_hits: bool = False, **kwargs):
        """
        custom encoder needed to handle serialisation issues with our data model
        :param kwargs: additional kwargs passed to json.dumps
        :return:
        """
        if drop_unmapped_ents or drop_hits:
            for section in self.sections:
                if drop_unmapped_ents:
                    ents_to_keep = list(filter(lambda x: x.mappings, section.entities))
                else:
                    ents_to_keep = section.entities
                if drop_hits:
                    for ent in ents_to_keep:
                        ent.hits.clear()
                section.entities = ents_to_keep

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
