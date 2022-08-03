import dataclasses
import json
import tempfile
import uuid
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import IntEnum, Enum
from itertools import cycle, chain
from math import inf
from typing import List, Any, Dict, Optional, Tuple, FrozenSet, Set, Iterable, Union

from kazu.utils.string_normalizer import StringNormalizer
from numpy import ndarray, float32, float16
from spacy import displacy

IS_SUBSPAN = "is_subspan"
# BIO schema
ENTITY_START_SYMBOL = "B"
ENTITY_INSIDE_SYMBOL = "I"
ENTITY_OUTSIDE_SYMBOL = "O"

# key for Document Processing Failed
PROCESSING_EXCEPTION = "PROCESSING_EXCEPTION"


class LinkRanks(IntEnum):
    # labels for ranking linking hits. NOTE! ordering important, as used for iteration
    HIGH_CONFIDENCE = 0
    MEDIUM_HIGH_CONFIDENCE = 1
    MEDIUM_CONFIDENCE = 2
    AMBIGUOUS = 3
    LOW_CONFIDENCE = 4


def remove_empty_elements(d):
    """recursively remove empty lists, empty dicts, or None elements from a dictionary"""

    def empty(x):
        return x is None or x == {} or x == []

    if not isinstance(d, (dict, list)):
        return d
    elif isinstance(d, list):
        return [v for v in (remove_empty_elements(v) for v in d) if not empty(v)]
    else:
        return {
            k: v for k, v in ((k, remove_empty_elements(v)) for k, v in d.items()) if not empty(v)
        }


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
    NO_STRATEGY = 1  # no strategy. should be used for debugging/testing only
    RESOLVED_BY_SIMILARITY = 2  # synonym linked to ID via similarity to default ID label
    SYNONYM_IS_AMBIGUOUS = 3  # synonym has no unambiguous meaning
    CUSTOM = 4  # a place holder for any strategy that
    UNAMBIGUOUS = 5


@dataclass(frozen=True, eq=True, order=True)
class EquivalentIdSet:
    """
    A representation of a set of kb ID's that map to the same synonym and mean the same thing.
    """

    ids: FrozenSet[str] = field(
        default_factory=frozenset, hash=True
    )  # other ID's mapping to this syn, from different KBs
    ids_to_source: Dict[str, str] = field(default_factory=dict, hash=False, compare=False)


@dataclass(frozen=True)
class Mapping:
    """
    a mapping is a fully mapped and disambiguated kb concept
    """

    default_label: str  # default label from knowledgebase
    source: str  # the knowledgebase/database/ontology name
    parser_name: str  # the origin of this mapping
    idx: str  # the identifier within the KB
    strategy: str  # the strategy used to create the mapping
    confidence: LinkRanks
    metadata: Dict[Any, Any] = field(default_factory=dict, hash=False)  # generic metadata


HitStoreKey = Tuple[str, EquivalentIdSet]  # parser name and EquivalentIdSet

NumericMetric = Union[bool, int, float]


@dataclass(frozen=True)
class SynonymTerm:
    terms: FrozenSet[str]
    term_norm: str
    parser_name: str
    is_symbolic: bool
    associated_id_sets: FrozenSet[EquivalentIdSet]
    mapping_types: FrozenSet[str] = field(hash=False)
    aggregated_by: EquivalentIdAggregationStrategy = field(hash=False, compare=False)

    @property
    def is_ambiguous(self):
        return len(self.associated_id_sets) > 1

    @property
    def is_confused(self):
        return self.is_ambiguous and len(self.terms) > 1


@dataclass(frozen=True, eq=True)
class Hit:
    """
    Hit is a precursor to a Mapping, meaning that a match has been detected and linked to an EquivalentIdSet, but
    is not ready to become a fully fledged mapping yet, as it may require further disambiguation
    """

    parser_name: str  # NOTE: this is the parser name, not the kb name.
    id_set: EquivalentIdSet
    # key is hit_string_norm, value is metric name to metric value
    per_normalized_syn_metrics: Dict[str, Dict[str, NumericMetric]] = field(
        default_factory=dict, hash=False
    )  # metrics associated with this hit

    def merge_metrics(self, hit: "Hit"):
        for hit_string_norm, new_metrics in hit.per_normalized_syn_metrics.items():
            existing_metrics = self.per_normalized_syn_metrics.get(hit_string_norm)
            if existing_metrics is None:
                self.per_normalized_syn_metrics[hit_string_norm] = new_metrics
            else:
                for metric_name, metric_val in new_metrics.items():
                    if metric_name in existing_metrics:
                        raise ValueError(
                            f"tried to add a metric {metric_name} that already exists on this hit"
                        )
                    existing_metrics[metric_name] = metric_val


@dataclass(frozen=True, eq=True)
class SynonymTermWithMetrics:
    terms: FrozenSet[str]
    term_norm: str
    parser_name: str
    is_symbolic: bool
    associated_id_sets: FrozenSet[EquivalentIdSet]
    aggregated_by: EquivalentIdAggregationStrategy
    mapping_types: FrozenSet[str] = field(compare=False)
    search_score: Optional[float] = field(compare=False, default=None)
    embed_score: Optional[float] = field(compare=False, default=None)
    bool_score: Optional[float] = field(compare=False, default=None)
    exact_match: Optional[bool] = field(compare=False, default=None)

    @staticmethod
    def from_synonym_term(
        term: SynonymTerm,
        search_score: Optional[float] = None,
        embed_score: Optional[float] = None,
        bool_score: Optional[float] = None,
        exact_match: Optional[bool] = None,
    ):

        return SynonymTermWithMetrics(
            search_score=search_score,
            embed_score=embed_score,
            bool_score=bool_score,
            exact_match=exact_match,
            **term.__dict__,
        )

    @property
    def is_ambiguous(self):
        return len(self.associated_id_sets) > 1

    @property
    def is_confused(self):
        return self.is_ambiguous and len(self.terms) > 1

    def merge_metrics(self, term: "SynonymTermWithMetrics") -> "SynonymTermWithMetrics":
        new_values = {
            k: v
            for k, v in term.__dict__.items()
            if k in {"search_score", "embed_score", "bool_score", "exact_match"} and v is not None
        }
        new_term = dataclasses.replace(self, **new_values)
        return new_term


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
    mappings: Set[Mapping] = field(default_factory=set)
    metadata: Dict[Any, Any] = field(default_factory=dict)  # generic metadata
    start: int = field(init=False)
    end: int = field(init=False)
    match_norm: str = field(init=False)
    syn_term_to_synonym_terms: Dict[SynonymTermWithMetrics, SynonymTermWithMetrics] = field(
        default_factory=dict
    )

    def update_terms(self, terms: Iterable[SynonymTermWithMetrics]):
        for term in terms:
            existing_term: Optional[SynonymTermWithMetrics] = self.syn_term_to_synonym_terms.get(
                term
            )
            if existing_term is not None:
                new_term = existing_term.merge_metrics(term)
                self.syn_term_to_synonym_terms[new_term] = new_term
            else:
                self.syn_term_to_synonym_terms[term] = term

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
        self.match_norm = StringNormalizer.normalize(self.match)

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
        self.mappings.add(mapping)

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

    metadata: Dict[Any, Any] = field(default_factory=dict, hash=False)  # generic metadata
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
        elif isinstance(obj, Entity):
            synonym_term_set = set(obj.syn_term_to_synonym_terms.values())
            as_dict = obj.__dict__
            # convert syn_term_to_synonym_terms to set
            as_dict.pop("syn_term_to_synonym_terms")
            as_dict["synonym_terms"] = synonym_term_set
            return as_dict
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

    @property
    def group_entities_on_hits(
        self,
    ) -> Iterable[Tuple[Tuple[str, str, FrozenSet[Hit]], Iterable[Entity]]]:
        yield from sort_then_group(self.get_entities(), key_func=_get_key_to_group_ent_on_hits)

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

    def as_minified_json(self, drop_unmapped_ents: bool = False, drop_hits: bool = False) -> str:
        as_dict_minified = self.as_minified_dict(drop_unmapped_ents, drop_hits)
        return json.dumps(as_dict_minified)

    def as_minified_dict(self, drop_unmapped_ents: bool = False, drop_hits: bool = False) -> Dict:
        as_dict = json.loads(self.json(drop_unmapped_ents, drop_hits))
        as_dict_minified = remove_empty_elements(as_dict)
        return as_dict_minified

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

    def __len__(self):
        length = 0
        for section in self.sections:
            length += len(section.get_text())
        return length


UNAMBIGUOUS_SYNONYM_MERGE_STRATEGIES = {EquivalentIdAggregationStrategy.UNAMBIGUOUS}
SimpleValue = Union[NumericMetric, str]
