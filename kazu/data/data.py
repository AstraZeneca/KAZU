import dataclasses
import json
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum, auto
from math import inf
from typing import List, Any, Dict, Optional, Tuple, FrozenSet, Set, Iterable, Union, DefaultDict

import bson
from bson import json_util
from numpy import ndarray, float32, float16

from kazu.utils.string_normalizer import StringNormalizer

IS_SUBSPAN = "is_subspan"
# BIO schema
ENTITY_START_SYMBOL = "B"
ENTITY_INSIDE_SYMBOL = "I"
ENTITY_OUTSIDE_SYMBOL = "O"

# key for Document Processing Failed
PROCESSING_EXCEPTION = "PROCESSING_EXCEPTION"


NoneType = type(None)
JsonDictType = Union[Dict[str, Any], List, int, float, bool, str, NoneType]


class AutoNameEnum(Enum):
    """Subclass to create an Enum where values are the names when using :class:`enum.auto`\\ .

    Taken from the `Python Enum Docs <https://docs.python.org/3/howto/enum.html#using-automatic-values>`_.

    This is `licensed under Zero-Clause BSD. <https://docs.python.org/3/license.html#zero-clause-bsd-license-for-code-in-the-python-release-documentation>`_

    .. raw:: html

        <details>
        <summary>Full License</summary>

    Permission to use, copy, modify, and/or distribute this software for any
    purpose with or without fee is hereby granted.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
    REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
    AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
    INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
    LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
    OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
    PERFORMANCE OF THIS SOFTWARE.

    .. raw:: html

        </details>
    """

    def _generate_next_value_(name, start, count, last_values):
        return name


class MentionConfidence(AutoNameEnum):
    HIGHLY_LIKELY = auto()  # almost certain to be correct
    POSSIBLE = auto()  # high degree of uncertainty


class StringMatchConfidence(AutoNameEnum):
    HIGHLY_LIKELY = auto()  # almost certain to be correct
    PROBABLE = auto()  # on the balance of probabilities, will be correct
    POSSIBLE = auto()  # high degree of uncertainty


class DisambiguationConfidence(AutoNameEnum):
    HIGHLY_LIKELY = auto()  # almost certain to be correct
    PROBABLE = auto()  # on the balance of probabilities, will be correct
    POSSIBLE = auto()  # high degree of uncertainty
    AMBIGUOUS = auto()  # could not disambiguate


@dataclass(frozen=True)
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


class EquivalentIdAggregationStrategy(AutoNameEnum):
    NO_STRATEGY = auto()  # no strategy. should be used for debugging/testing only
    RESOLVED_BY_SIMILARITY = auto()  # synonym linked to ID via similarity to default ID label
    SYNONYM_IS_AMBIGUOUS = auto()  # synonym has no unambiguous meaning
    CUSTOM = auto()  # a place holder for any strategy that
    UNAMBIGUOUS = auto()
    MERGED_AS_NON_SYMBOLIC = auto()  # used when non-symbolic synonyms are merged
    MODIFIED_BY_CURATION = auto()


@dataclass(frozen=True, eq=True, order=True)
class EquivalentIdSet:
    """
    A representation of a set of kb ID's that map to the same synonym and mean the same thing.
    """

    ids_and_source: FrozenSet[Tuple[str, str]] = field(
        default_factory=frozenset, hash=True
    )  # other ID's mapping to this syn, from different KBs

    @property
    def sources(self) -> Set[str]:
        return set(x[1] for x in self.ids_and_source)

    @property
    def ids(self) -> Set[str]:
        return set(x[0] for x in self.ids_and_source)


@dataclass(frozen=True)
class Mapping:
    """
    a mapping is a fully mapped and disambiguated kb concept
    """

    default_label: str  # default label from knowledgebase
    source: str  # the knowledgebase/database/ontology name
    parser_name: str  # the origin of this mapping
    idx: str  # the identifier within the KB
    string_match_strategy: str
    string_match_confidence: StringMatchConfidence
    disambiguation_confidence: Optional[DisambiguationConfidence] = None
    disambiguation_strategy: Optional[str] = None
    xref_source_parser_name: Optional[str] = None  # source parser name if mapping is an XREF
    metadata: Dict[Any, Any] = field(default_factory=dict, hash=False)  # generic metadata


NumericMetric = Union[bool, int, float]

AssociatedIdSets = FrozenSet[EquivalentIdSet]


@dataclass(frozen=True, eq=True)
class SynonymTerm:
    """
    a SynonymTerm is a container for a single normalised synonym, and is produced by an OntologyParser implementation.
    It may be composed of multiple terms that normalise
    to the same unique string (e.g. "breast cancer" and "Breast Cancer"). The number of associated_id_sets that this
    synonym maps to is determined by the  score_and_group_ids method of the associated OntologyParser
    """

    terms: FrozenSet[str]  # unnormalised synonym strings
    term_norm: str  # normalised form
    parser_name: str  # ontology parser name
    is_symbolic: bool  # is the term symbolic? Determined by the OntologyParser
    associated_id_sets: AssociatedIdSets
    mapping_types: FrozenSet[str] = field(hash=False)  # mapping type metadata
    aggregated_by: EquivalentIdAggregationStrategy = field(
        hash=False, compare=False
    )  # determined by the ontology parser
    original_term: Optional[
        str
    ] = None  # if generated by a synonym generator, the original source synonym

    @property
    def is_ambiguous(self):
        return len(self.associated_id_sets) > 1


@dataclass(frozen=True, eq=True)
class SynonymTermWithMetrics(SynonymTerm):
    """
    Similar to SynonymTerm, but also allows metrics to be scored. As these metrics are not used in the hash function,
    care should be taken when hashing of this object is required
    """

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
    A :class:`kazu.data.data.Entity` is a container for information about a single entity detected within a :class:`kazu.data.data.Section`

    Within an :class:`kazu.data.data.Entity`, the most important fields are :attr:`.Entity.match` (the actual string detected),
    :attr:`.Entity.syn_term_to_synonym_terms`, a dict of :class:`kazu.data.data.SynonymTermWithMetrics` (candidates for knowledgebase hits)
    and :attr:`.Entity.mappings`, the final product of linked references to the underlying entity

    """

    match: str  # exact text representation
    entity_class: str  # entity class
    spans: FrozenSet[CharSpan]  # charspans
    namespace: str  # namespace of Step that produced this instance
    mention_confidence: MentionConfidence = MentionConfidence.HIGHLY_LIKELY
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
        self.match_norm = StringNormalizer.normalize(self.match, self.entity_class)

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
    _sentence_spans: Optional[Tuple[CharSpan, ...]] = field(
        default=None, hash=False, init=False
    )  # hidden implem. to prevent overwriting existing sentence spans

    @property
    def sentence_spans(self) -> Iterable[CharSpan]:
        if self._sentence_spans is not None:
            return self._sentence_spans
        else:
            return ()

    @sentence_spans.setter
    def sentence_spans(self, sent_spans: Iterable[CharSpan]):
        """
        Setter for sentence_spans. sentence_spans are stored in the order provided by the iterable
        sent_spans param, which may not necessarily be in sorted order.

        :param sent_spans:
        :return:
        """
        if self._sentence_spans is None:
            sent_spans_tuple = tuple(sent_spans)
            assert len(sent_spans_tuple) == len(
                set(sent_spans_tuple)
            ), "There are duplicate sentence spans"
            self._sentence_spans = sent_spans_tuple
        else:
            raise AttributeError("Immutable sentence_spans is already set")

    def __str__(self):
        return f"name: {self.name}, text: {self.get_text()[:100]}"

    def get_text(self) -> str:
        """
        rather than accessing text or preprocessed_text directly, this method provides a convenient wrapper to get
        preprocessed_text if available, or text if not.
        """
        if self.preprocessed_text is None:
            return self.text
        else:
            return self.preprocessed_text


@dataclass(unsafe_hash=True)
class Document:
    idx: str = field(default_factory=lambda: uuid.uuid4().hex)  # a document identifier
    sections: List[Section] = field(
        default_factory=list, hash=False
    )  # sections comprising this document
    metadata: Dict[Any, Any] = field(default_factory=dict, hash=False)  # generic metadata

    def __str__(self):
        return f"idx: {self.idx}"

    def get_entities(self) -> List[Entity]:
        """
        get all entities in this document
        """
        entities = []
        for section in self.sections:
            entities.extend(section.entities)
        return entities

    def json(
        self,
        drop_unmapped_ents: bool = False,
        drop_terms: bool = False,
        **kwargs,
    ):
        """
        custom encoder needed to handle serialisation issues with our data model

        :param: drop_unmapped_ents: drop any entities that have no mappings
        :param: drop_terms: drop the synonym term dict field
        :param kwargs: additional kwargs passed to json.dumps
        :return:
        """
        as_dict = self.as_minified_dict(
            drop_unmapped_ents=drop_unmapped_ents, drop_terms=drop_terms
        )
        return json.dumps(as_dict, **kwargs)

    def as_minified_dict(self, drop_unmapped_ents: bool = False, drop_terms: bool = False) -> Dict:
        as_dict = DocumentJsonUtils.doc_to_json_dict(self)
        as_dict = DocumentJsonUtils.minify_json_dict(
            as_dict, drop_unmapped_ents=drop_unmapped_ents, drop_terms=drop_terms
        )
        return DocumentJsonUtils.remove_empty_elements(as_dict)

    @classmethod
    def create_simple_document(cls, text: str) -> "Document":
        """
        Create an instance of :class:`.Document` from a text string.

        :param text:
        :return:
        """
        return cls(sections=[Section(text=text, name="na")])

    @classmethod
    def simple_document_from_sents(cls, sents: List[str]) -> "Document":
        section = Section(text=" ".join(sents), name="na")
        sent_spans = []
        curr_start = 0
        for sent in sents:
            sent_spans.append(CharSpan(start=curr_start, end=curr_start + len(sent)))
            curr_start += len(sent) + 1  # + 1 is for the joining space
        section.sentence_spans = sent_spans
        return cls(sections=[section])

    @classmethod
    def from_named_section_texts(cls, named_sections: Dict[str, str]) -> "Document":
        sections = [Section(text=text, name=name) for name, text in named_sections.items()]
        return cls(sections=sections)

    def __len__(self):
        length = 0
        for section in self.sections:
            length += len(section.get_text())
        return length


class DocumentJsonUtils:
    class ConversionException(Exception):
        pass

    atomic_types = (int, float, str, bool, type(None))
    listlike_types = (list, tuple, set, frozenset)

    @staticmethod
    def minify_json_dict(
        doc_json_dict: Dict[str, Any],
        drop_unmapped_ents: bool = False,
        drop_terms: bool = False,
        in_place=True,
    ) -> Dict:
        doc_json_dict = doc_json_dict if in_place else deepcopy(doc_json_dict)

        if drop_unmapped_ents or drop_terms:
            for section_dict in doc_json_dict["sections"]:
                section_entities = section_dict["entities"]
                ents_to_keep = list(
                    filter(lambda _ent: _ent["mappings"], section_entities)
                    if drop_unmapped_ents
                    else section_entities
                )
                if drop_terms:
                    for ent in ents_to_keep:
                        ent["synonym_terms"].clear()

                section_dict["entities"] = ents_to_keep

        return doc_json_dict

    @classmethod
    def doc_to_json_dict(cls, doc: Document) -> Dict[str, JsonDictType]:
        return {k: DocumentJsonUtils.obj_to_dict_repr(v) for k, v in doc.__dict__.items()}

    @classmethod
    def obj_to_dict_repr(cls, obj: Any) -> JsonDictType:
        if isinstance(obj, cls.atomic_types):
            return obj
        elif isinstance(obj, (float16, float32)):  # type: ignore[misc]
            return obj.item()
        elif isinstance(obj, cls.listlike_types):
            return [cls.obj_to_dict_repr(elem) for elem in obj]
        elif isinstance(obj, ndarray):
            return obj.tolist()
        elif dataclasses.is_dataclass(obj):
            return cls.obj_to_dict_repr(obj.__dict__)
        elif isinstance(obj, dict):
            processed_dict_pairs = (cls._preprocess_dict_pair(pair) for pair in obj.items())
            return {k: cls.obj_to_dict_repr(v) for k, v in processed_dict_pairs}
        elif isinstance(obj, Enum):
            return obj.name
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, bson.ObjectId):
            return json_util.default(obj)
        else:
            raise cls.ConversionException(f"Unknown object type: {type(obj)}")

    @classmethod
    def _preprocess_dict_pair(cls, kv_pair: Tuple[Any, Any]) -> Tuple[str, JsonDictType]:
        k, v = kv_pair
        if k == "offset_map":
            # v is a dict of CharSpan->CharSpan, needs conversion
            return k, list(v.items())
        elif k == "_sentence_spans":
            return "sentence_spans", list(v)
        elif k == "syn_term_to_synonym_terms":
            return "synonym_terms", list(v.values())
        else:
            return k, v

    @staticmethod
    def remove_empty_elements(d: Any):
        """recursively remove empty lists, empty dicts, or None elements from a dictionary"""
        if not isinstance(d, (dict, list)):
            return d
        elif isinstance(d, list):
            return [
                v
                for v in (DocumentJsonUtils.remove_empty_elements(v) for v in d)
                if not DocumentJsonUtils.empty(v)
            ]
        else:
            return {
                k: v
                for k, v in ((k, DocumentJsonUtils.remove_empty_elements(v)) for k, v in d.items())
                if not DocumentJsonUtils.empty(v)
            }

    @staticmethod
    def empty(x) -> bool:
        return x is None or x == {} or x == []


SimpleValue = Union[NumericMetric, str]


class SynonymTermBehaviour(AutoNameEnum):
    #: do nothing. Useful if you want to catalogue a term as "I've looked at this but don't want to use it for NER"
    IGNORE = auto()
    #: use the term for both dictionary based NER and as a linking target.
    ADD_FOR_NER_AND_LINKING = auto()
    #: use the term only as a linking target. Note, this is not required if the term is already in the
    #: underlying ontology, as all ontology terms are included as linking targets by default
    #: (Also see DROP_SYNONYM_TERM_FOR_LINKING)
    ADD_FOR_LINKING_ONLY = auto()
    #: do not use this term as a linking target. Normally, you would use this for a term you want to remove
    #: from the underlying ontology (e.g. a 'bad' synonym). If the term does not exist, has no effect
    DROP_SYNONYM_TERM_FOR_LINKING = auto()
    #: removes an EquivalentIdSet from the term. The EquivalentIdSet to remove is selected via being a
    #: superset of the IDs in the associated parser_to_target_id_mappings value
    DROP_ID_SET_FROM_SYNONYM_TERM = auto()


class ParserBehaviour(AutoNameEnum):
    #: completely remove the ids from the parser - i.e. should never used anywhere
    DROP_IDS_FROM_PARSER = auto()


@dataclass(frozen=True, eq=True)
class SynonymTermAction:
    """
    A SynonymTermAction is an action that affects the :class:`.SynonymTerm`\\s that a parser
    produces.

    Note, the term_norm field is calculated by the :class:`kazu.modelling.ontology_preprocessing.base.OntologyParser`
    if required, and should not be set manually.

    The parser_to_target_id_mappings field should specify the parser name and a set of affected ID's if required.

    See :class:`.SynonymTermBehaviour` for the type of actions that are possible.

    """

    behaviour: SynonymTermBehaviour
    associated_id_sets: Optional[AssociatedIdSets] = None

    def __post_init__(self):
        if (
            self.behaviour is not SynonymTermBehaviour.IGNORE
            and self.behaviour is not SynonymTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING
            and self.behaviour is not SynonymTermBehaviour.INHERIT_FROM_SOURCE_TERM
        ) and (self.associated_id_sets is None or len(self.associated_id_sets) == 0):
            raise ValueError(f"associated_id_sets must be specified for behaviour {self.behaviour}")

    @classmethod
    def from_dict(cls, json_dict: Dict) -> "SynonymTermAction":
        if json_dict["associated_id_sets"] is not None:
            assoc_id_sets = set()
            for equiv_id_set in json_dict["associated_id_sets"]:
                new_equiv_id_lst = []
                for equiv_id_lst in equiv_id_set["ids_and_source"]:
                    idx = equiv_id_lst[0]
                    source = equiv_id_lst[1]
                    new_equiv_id_lst.append(
                        (
                            idx,
                            source,
                        )
                    )
                assoc_id_sets.add(EquivalentIdSet(frozenset(new_equiv_id_lst)))

            frozen_assoc_id_sets = frozenset(assoc_id_sets)
        else:
            frozen_assoc_id_sets = None
        instance = cls(
            behaviour=SynonymTermBehaviour(json_dict["behaviour"]),
            associated_id_sets=frozen_assoc_id_sets,
        )
        return instance


@dataclass(frozen=True)
class ParserAction:
    """
    A ParserAction changes the behaviour of a :class:`kazu.modelling.ontology_preprocessing.base.OntologyParser` in
    a global sense.

    A ParserAction overrides any default behaviour of the parser, and also any conflicts that may occur with
    :class:`.SynonymTermAction`\\s.

    These actions are useful for eliminating unwanted behaviour. For example, the root of the Mondo
    ontology is http://purl.obolibrary.org/obo/HP_0000001, which has a default label of 'All'. Since this is
    such a common word, and not very useful in terms of linking, we might want a global action so that this
    ID is not used anywhere in a Kazu pipeline.

    The parser_to_target_id_mappings field should specify the parser name and an affected IDs if required.
    See :class:`.ParserBehaviour` for the type of actions that are possible.

    """

    behaviour: ParserBehaviour
    parser_to_target_id_mappings: Dict[str, Set[str]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_dict: Dict) -> "ParserAction":
        return cls(
            behaviour=ParserBehaviour(json_dict["behaviour"]),
            parser_to_target_id_mappings={
                k: set(v) for k, v in json_dict["parser_to_target_id_mappings"].items()
            },
        )

    def __post_init__(self):
        if len(self.parser_to_target_id_mappings) == 0:
            raise ValueError(
                f"parser_to_target_id_mappings must be specified for global actions: {self}"
            )
        for key, values in self.parser_to_target_id_mappings.items():
            if len(values) == 0:
                raise ValueError(f"at least one ID must be specified for key: {key}, {self}")


@dataclass(frozen=True)
class GlobalParserActions:
    """
    Container for all :class:`.ParserAction`\\s.

    """

    actions: List[ParserAction]
    _parser_name_to_action: DefaultDict[str, List[ParserAction]] = field(
        default_factory=lambda: defaultdict(list), init=False
    )

    def __post_init__(self):
        for action in self.actions:
            for parser_name in action.parser_to_target_id_mappings.keys():
                self._parser_name_to_action[parser_name].append(action)

    def parser_behaviour(self, parser_name: str) -> Iterable[ParserAction]:
        """
        Generator that yields behaviours for a specific parser, based on the order they are
        specified in.

        :param parser_name:
        :return:
        """
        yield from self._parser_name_to_action.get(parser_name, [])

    @classmethod
    def from_json(cls, json_dict: Dict) -> "GlobalParserActions":
        return cls(
            actions=[ParserAction.from_json(x) for x in json_dict["actions"]],
        )


@dataclass(frozen=True, eq=True)
class Curation:
    """
    A Curation is a means to modify the behaviour of a specific :class:`.SynonymTerm`.

    This can affect both the behaviour of :class:`kazu.modelling.ontology_preprocessing.base.OntologyParser`,
    and dictionary based NER (if using the :class:`kazu.steps.joint_ner_and_linking.explosion.ExplosionStringMatchingStep`).
    This is controlled by the list of :class:`.SynonymTermAction`\\s attached.

    Example 1:

    The string 'ALL' is highly ambiguous. It might mean several diseases, or simply 'all'. Therefore, we want
    to add a curation as follows, so that it will only be used as a linking target and not for dictionary based NER:

    .. code-block:: python

        Curation(
            curated_synonym="ALL",
            case_sensitive=True,
            actions=tuple([
                SynonymTermAction(
                    behaviour=SynonymTermBehaviour.ADD_FOR_LINKING_ONLY,
                    parser_to_target_id_mappings={"OPENTARGETS_DISEASE": {"MONDO:0004967"}},
                    entity_class="disease",
                ),
            ]),
            mention_confidence=MentionConfidence.POSSIBLE,
        )


    Example 2:

    The string 'LH' is incorrectly identified as a synonym of the PLOD1 (ENSG00000083444) gene, whereas more often than not, it's actually an abbreviation of Lutenising Hormone.
    We therefore want to drop this reference, and add a new one to LHB (ENSG00000104826, or Lutenising Hormone Subunit Beta)

    The Curation we therefore want is:

    .. code-block:: python

        Curation(
            curated_synonym="LH",
            case_sensitive=True,
            actions=tuple([
                SynonymTermAction(
                    behaviour=SynonymTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING,
                    parser_to_target_id_mappings={"OPENTARGETS_TARGET": set()},
                    entity_class="gene",
                ),
                SynonymTermAction(
                    behaviour=SynonymTermBehaviour.ADD_FOR_NER_AND_LINKING,
                    parser_to_target_id_mappings={"OPENTARGETS_TARGET": {"ENSG00000104826"}},
                    entity_class="gene",
                ),
            ]),
            mention_confidence=MentionConfidence.POSSIBLE,
        )

    Example 3:

    A :class:`.SynonymTerm` has an unwanted :class:`.EquivalentIdSet` attached to it, but is otherwise good. We want to remove this set

    .. code-block:: python

        Curation(
            curated_synonym="some good synonym",
            case_sensitive=True,
            actions=tuple([
                SynonymTermAction(
                    behaviour=SynonymTermBehaviour.DROP_ID_SET_FROM_SYNONYM_TERM,
                    parser_to_target_id_mappings={
                        "OPENTARGETS_TARGET": {"an id from the bad set"}
                    },
                    entity_class="gene",
                ),
            ]),
            mention_confidence=MentionConfidence.POSSIBLE,
        )

    Notes:

    The term_norm field is calculated by the :class:`kazu.modelling.ontology_preprocessing.base.OntologyParser`
    if required, and should not be set manually.

    mention_confidence is currently non-functional, and will be expanded upon at a later date



    """

    curated_synonym: str
    mention_confidence: MentionConfidence
    actions: Tuple[SynonymTermAction, ...]
    case_sensitive: bool
    _id: bson.ObjectId = field(default_factory=bson.ObjectId)
    # the original term that is used as a 'seed' term for this curation.
    # note, this is used for NER to determine how linking is performed. If you also
    # want to use this curation as a linking target for non-dictionary based NER processes,
    # or the term is identical to the one used in the source ontology,
    # this should be set to None, so that a novel term_norm is calculated
    source_term: Optional[str] = None

    def __post_init__(self):
        # data validation
        if not isinstance(self.curated_synonym, str):
            raise ValueError(f"curated_synonym should be a string, {self}")

    def source_term_norm(self, entity_class: str):
        return StringNormalizer.normalize(self.source_term, entity_class)

    def curated_synonym_norm(self, entity_class: str):
        return StringNormalizer.normalize(self.curated_synonym, entity_class)

    def term_norm_for_linking(self, entity_class: str):
        norm_target = self.curated_synonym if self.source_term is None else self.source_term
        return StringNormalizer.normalize(norm_target, entity_class)

    @classmethod
    def from_json(cls, json_str: str) -> "Curation":
        json_dict = json_util.loads(json_str)
        return cls.from_dict(json_dict)

    @classmethod
    def from_dict(cls, json_dict: Dict) -> "Curation":

        return cls(
            mention_confidence=MentionConfidence(json_dict["mention_confidence"]),
            actions=tuple([SynonymTermAction.from_dict(x) for x in json_dict["actions"]]),
            case_sensitive=json_dict["case_sensitive"],
            curated_synonym=json_dict["curated_synonym"],
            source_term=json_dict["source_term"],
            _id=json_dict.get("_id", bson.ObjectId()),
        )

    def to_dict(self, preserve_object_id: bool = True):
        if preserve_object_id:
            oid = self._id
            as_dict = DocumentJsonUtils.obj_to_dict_repr(self)
            as_dict["_id"] = oid  # type: ignore[index,call-overload]
        else:
            as_dict = DocumentJsonUtils.obj_to_dict_repr(self)
        return as_dict

    def to_json(self) -> str:
        as_json = self.to_dict(False)
        assert isinstance(as_json, dict)
        return json.dumps(as_json)
