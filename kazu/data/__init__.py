"""This module contains the core aspects of the :doc:`/datamodel`.

See the page linked above for a quick introduction to the key concepts.

.. |metadata_s11n_warn| replace::

   Note that storing objects here that Kazu can't
   convert to and from json will cause problems for (de)serialization.
   See :ref:`deserialize-generic-metadata` for details.
"""

import json
import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto, IntEnum
from math import inf
from typing import Any, Optional, Union

import bson
import cattrs.preconf.json
import cattrs.strategies
from bson import json_util
from kazu.utils.string_normalizer import StringNormalizer
from numpy import ndarray, float32, float16

IS_SUBSPAN = "is_subspan"
# BIO schema
ENTITY_START_SYMBOL = "B"
ENTITY_INSIDE_SYMBOL = "I"
ENTITY_OUTSIDE_SYMBOL = "O"

# key for Document Processing Failed
PROCESSING_EXCEPTION = "PROCESSING_EXCEPTION"


NumericMetric = Union[bool, int, float]
SimpleValue = Union[NumericMetric, str]
JsonEncodable = Optional[Union[dict[str, "JsonEncodable"], list["JsonEncodable"], SimpleValue]]
"""Represents a json-encodable object.

Note that because :class:`dict` is invariant, there can be issues with
using types like ``dict[str, str]`` (see further
`here <https://github.com/python/typing/issues/182#issuecomment-1320974824>`_).
"""


class AutoNameEnum(Enum):
    """Subclass to create an Enum where values are the names when using
    :class:`enum.auto`\\ .

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


class MentionConfidence(IntEnum):
    HIGHLY_LIKELY = 100  # almost certain to be correct
    PROBABLE = 50
    POSSIBLE = 10  # high degree of uncertainty
    IGNORE = 0  # do not use this mention for NER


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
    """A concept similar to a spaCy Span, except is character index based rather than
    token based.

    Example: ``text[char_span.start:char_span.end]`` will precisely cover the target.
    This means that ``text[char_span.end]`` will give the first character **not** in the
    span.
    """

    start: int
    end: int

    def is_completely_overlapped(self, other: "CharSpan") -> bool:
        """True if other completely overlaps this span.

        :param other:
        :return:
        """
        return self.start >= other.start and self.end <= other.end

    def is_partially_overlapped(self, other: "CharSpan") -> bool:
        """True if other partially overlaps this span.

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
    RESOLVED_BY_XREF = auto()


# this is frozen below, but for use in typehints elsewhere the mutable version is
# more useful, as a function written for an immutable frozenset would often behave
# correctly when passed a mutable set, and sometimes our functions actually produce
# the mutable version, as all we wish to do is compare with an immutable version,
# and the mutable version is easier to produce when constructed iteratively.
IdsAndSource = set[tuple[str, str]]


@dataclass(frozen=True, eq=True, order=True)
class EquivalentIdSet:
    """A representation of a set of kb ID's that map to the same synonym and mean the
    same thing."""

    # other ID's mapping to this syn, from different KBs
    ids_and_source: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    @property
    def sources(self) -> set[str]:
        return set(x[1] for x in self.ids_and_source)

    @property
    def ids(self) -> set[str]:
        return set(x[0] for x in self.ids_and_source)


@dataclass(frozen=True)
class Mapping:
    """A mapping is a fully mapped and disambiguated kb concept."""

    #: default label from knowledgebase
    default_label: str
    #: the knowledgebase/database/ontology name
    source: str
    #: the origin of this mapping
    parser_name: str
    #: the identifier within the KB
    idx: str
    string_match_strategy: str
    string_match_confidence: StringMatchConfidence
    disambiguation_confidence: Optional[DisambiguationConfidence] = None
    disambiguation_strategy: Optional[str] = None
    #: source parser name if mapping is an XREF
    xref_source_parser_name: Optional[str] = None
    #: | generic metadata
    #: |
    #: | |metadata_s11n_warn|
    metadata: dict = field(default_factory=dict, hash=False)

    @staticmethod
    def from_dict(mapping_dict: dict) -> "Mapping":
        return kazu_json_converter.structure(mapping_dict, Mapping)


AssociatedIdSets = frozenset[EquivalentIdSet]
"""A frozen set of :class:`.EquivalentIdSet`"""


@dataclass(frozen=True, eq=True)
class LinkingCandidate:
    """A LinkingCandidate is a container for a single normalised synonym, and is
    produced by an :class:`~.OntologyParser` implementation.

    It may be composed of multiple synonyms that normalise to the same
    unique string (e.g. "breast cancer" and "Breast Cancer"). The number
    of ``associated_id_sets`` that this synonym maps to is determined by the
    :meth:`~.OntologyParser.score_and_group_ids` method of the associated OntologyParser.
    """

    #: unnormalised synonym strings
    raw_synonyms: frozenset[str]
    #: normalised form
    synonym_norm: str
    #: ontology parser name
    parser_name: str
    #: is the candidate symbolic? Determined by the OntologyParser
    is_symbolic: bool
    associated_id_sets: AssociatedIdSets
    #: aggregation strategy, determined by the ontology parser
    aggregated_by: EquivalentIdAggregationStrategy = field(hash=False, compare=False)
    #: mapping type metadata
    mapping_types: frozenset[str] = field(hash=False, default_factory=frozenset)

    @property
    def is_ambiguous(self) -> bool:
        return len(self.associated_id_sets) > 1

    @staticmethod
    def from_dict(candidate_dict: dict) -> "LinkingCandidate":
        return kazu_json_converter.structure(candidate_dict, LinkingCandidate)


@dataclass()
class LinkingMetrics:
    """Metrics for Entity Linking.

    LinkingMetrics holds data on various quality metrics, for how well a :class:`~.LinkingCandidate`
     maps to a host :class:`~.Entity`\\.
    """

    search_score: Optional[float] = field(compare=False, default=None)
    embed_score: Optional[float] = field(compare=False, default=None)
    bool_score: Optional[bool] = field(compare=False, default=None)
    exact_match: Optional[bool] = field(compare=False, default=None)

    @staticmethod
    def from_dict(metric_dict: dict) -> "LinkingMetrics":
        return kazu_json_converter.structure(metric_dict, LinkingMetrics)


CandidatesToMetrics = dict[LinkingCandidate, LinkingMetrics]
"""This type is used whenever we have :class:`~.LinkingCandidate`\\ s and some metrics
for how well they map to a specific :class:`~.Entity`\\.

In particular, :attr:`~.Entity.linking_candidates` holds relevant candidates and their metrics, and this type is used in parts
of kazu which produce, modify or use this field.
"""


@dataclass(unsafe_hash=True)
class Entity:
    """A :class:`kazu.data.Entity` is a container for information about a
    single entity detected within a :class:`kazu.data.Section`\\ .

    Within an :class:`kazu.data.Entity`, the most important fields are :attr:`.Entity.match` (the actual string detected),
    :attr:`.Entity.linking_candidates`, (candidates for knowledgebase hits)
    and :attr:`.Entity.mappings`, the final product of linked references to the underlying entity.
    """

    #: exact text representation
    match: str
    entity_class: str
    spans: frozenset[CharSpan]
    #: namespace of the :class:`~.Step` that produced this instance
    namespace: str
    mention_confidence: MentionConfidence = MentionConfidence.HIGHLY_LIKELY
    _id: str = field(default_factory=lambda: uuid.uuid4().hex)
    mappings: set[Mapping] = field(default_factory=set, hash=False)
    #: | generic metadata
    #: |
    #: | |metadata_s11n_warn|
    metadata: dict = field(default_factory=dict, hash=False)
    start: int = field(init=False, hash=False)
    end: int = field(init=False, hash=False)
    match_norm: str = field(init=False, hash=False)
    linking_candidates: CandidatesToMetrics = field(default_factory=dict, hash=False)

    def add_or_update_linking_candidates(self, candidates: CandidatesToMetrics) -> None:
        for candidate, metrics in candidates.items():
            self.add_or_update_linking_candidate(candidate, metrics)

    def add_or_update_linking_candidate(
        self, candidate: LinkingCandidate, new_metrics: LinkingMetrics
    ) -> None:
        maybe_existing_metrics = self.linking_candidates.get(candidate)
        if not maybe_existing_metrics:
            self.linking_candidates[candidate] = new_metrics
        else:
            for k, v in new_metrics.__dict__.items():
                if v is not None:
                    setattr(maybe_existing_metrics, k, v)

    def calc_starts_and_ends(self) -> tuple[int, int]:
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

    def is_completely_overlapped(self, other: "Entity") -> bool:
        """True if all CharSpan instances are completely encompassed by all other
        CharSpan instances.

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

    def is_partially_overlapped(self, other: "Entity") -> bool:
        """True if only one CharSpan instance is defined in both self and other, and
        they are partially overlapped.

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
        lung cancer -> [CharSpan(0,4), CharSpan(15, 21)]
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
        """Span length.

        :return: number of characters enclosed by span
        """
        return self.end - self.start

    def __repr__(self) -> str:
        """Describe the tag.

        :return: tag match description
        """
        return f"{self.match}:{self.entity_class}:{self.namespace}:{self.start}:{self.end}"

    def as_brat(self) -> str:
        """
        :return: this entity in the third party biomedical nlp Brat format
            (see the `docs <https://brat.nlplab.org/introduction.html>`_\\ ,
            `paper <https://aclanthology.org/E12-2021.pdf>`_\\ , and
            `codebase <https://github.com/nlplab/brat>`_\\ )
        """
        # TODO: update this to make use of non-contiguous entities
        return f"{hash(self)}\t{self.entity_class}\t{self.start}\t{self.end}\t{self.match}\n"

    def add_mapping(self, mapping: Mapping) -> None:
        """Deprecated.

        :param mapping:
        :return:
        """
        self.mappings.add(mapping)

    @classmethod
    def from_spans(
        cls, spans: list[tuple[int, int]], text: str, join_str: str = "", **kwargs: Any
    ) -> "Entity":
        """Create an instance of Entity from a list of character indices. A text string
        of underlying doc is also required to produce a representative match.

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
    def load_contiguous_entity(cls, start: int, end: int, **kwargs: Any) -> "Entity":
        single_span = frozenset([CharSpan(start=start, end=end)])
        return cls(spans=single_span, **kwargs)

    @staticmethod
    def from_dict(entity_dict: dict) -> "Entity":
        return kazu_json_converter.structure(entity_dict, Entity)


@dataclass(unsafe_hash=True)
class Section:
    """A container for text and entities.

    One or more make up a :class:`kazu.data.Document`.
    """

    #: the text to be processed
    text: str
    #: the name of the section (e.g. abstract, body, header, footer etc)
    name: str
    #: | generic metadata
    #: |
    #: | |metadata_s11n_warn|
    metadata: dict = field(default_factory=dict, hash=False)
    #: entities detected in this section
    entities: list[Entity] = field(default_factory=list, hash=False)
    _sentence_spans: Optional[tuple[CharSpan, ...]] = field(
        default=None, hash=False, init=False
    )  # hidden implem. to prevent overwriting existing sentence spans

    @property
    def sentence_spans(self) -> Iterable[CharSpan]:
        if self._sentence_spans is not None:
            return self._sentence_spans
        else:
            return ()

    @sentence_spans.setter
    def sentence_spans(self, sent_spans: Iterable[CharSpan]) -> None:
        """Setter for sentence_spans. sentence_spans are stored in the order provided by
        the iterable sent_spans param, which may not necessarily be in sorted order.

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
        return f"name: {self.name}, text: {self.text[:100]}"

    @staticmethod
    def from_dict(section_dict: dict) -> "Section":
        return kazu_json_converter.structure(section_dict, Section)


@dataclass(unsafe_hash=True)
class Document:
    """A container that is the primary input into a :class:`kazu.pipeline.Pipeline`."""

    #: a document identifier
    idx: str = field(default_factory=lambda: uuid.uuid4().hex)
    #: sections comprising this document
    sections: list[Section] = field(default_factory=list, hash=False)
    #: | generic metadata
    #: |
    #: | |metadata_s11n_warn|
    metadata: dict = field(default_factory=dict, hash=False)

    def __str__(self):
        return f"idx: {self.idx}"

    def get_entities(self) -> list[Entity]:
        """Get all entities in this document."""
        entities = []
        for section in self.sections:
            entities.extend(section.entities)
        return entities

    def to_json(
        self,
        **kwargs: Any,
    ) -> str:
        """Convert to json string.

        :param kwargs: passed through to :func:`json.dumps`.
        :return:
        """
        return kazu_json_converter.dumps(self, **kwargs)

    def to_dict(self) -> dict:
        """Convert the Document to a ``dict``."""
        # type ignore needed because cattrs says this could be 'any', but we know more specifically it will be a
        # json-encodable dict
        return kazu_json_converter.unstructure(self)  # type: ignore[no-any-return]

    @classmethod
    def create_simple_document(cls, text: str) -> "Document":
        """Create an instance of :class:`.Document` from a text string.

        :param text:
        :return:
        """
        return cls(sections=[Section(text=text, name="na")])

    @classmethod
    def simple_document_from_sents(cls, sents: list[str]) -> "Document":
        section = Section(text=" ".join(sents), name="na")
        sent_spans = []
        curr_start = 0
        for sent in sents:
            sent_spans.append(CharSpan(start=curr_start, end=curr_start + len(sent)))
            curr_start += len(sent) + 1  # + 1 is for the joining space
        section.sentence_spans = sent_spans
        return cls(sections=[section])

    @classmethod
    def from_named_section_texts(cls, named_sections: dict[str, str]) -> "Document":
        sections = [Section(text=text, name=name) for name, text in named_sections.items()]
        return cls(sections=sections)

    def __len__(self):
        return sum(len(section.text) for section in self.sections)

    @staticmethod
    def from_dict(document_dict: dict) -> "Document":
        return kazu_json_converter.structure(document_dict, Document)

    @staticmethod
    def from_json(json_str: str) -> "Document":
        return Document.from_dict(json.loads(json_str))


def _initialize_json_converter(testing: bool = False) -> cattrs.preconf.json.JsonConverter:
    json_conv = cattrs.preconf.json.make_converter(omit_if_default=True, forbid_extra_keys=True)

    # 'external' to kazu datatypes
    json_conv.register_unstructure_hook(float16, lambda v: v.item())
    json_conv.register_structure_hook(float16, lambda v, _: float16(v))
    json_conv.register_unstructure_hook(float32, lambda v: v.item())
    json_conv.register_structure_hook(float32, lambda v, _: float16(v))
    json_conv.register_unstructure_hook(ndarray, lambda v: v.tolist())
    # note: this could result in a change in the dtype (e.g. float vs int and precision)
    # from the 'original' if roundtripped. This doesn't seem like a deal breaker for
    # what we're using it for.
    json_conv.register_structure_hook(ndarray, lambda v, _: ndarray(v))
    json_conv.register_unstructure_hook(bson.ObjectId, lambda v: json_util.default(v))
    json_conv.register_structure_hook(bson.ObjectId, lambda v, _: json_util.object_hook(v))

    json_conv.register_unstructure_hook(MentionConfidence, lambda v: v.name)
    json_conv.register_structure_hook(MentionConfidence, lambda v, _: MentionConfidence[v])

    def _linking_candidates_unstruct_hook(
        candidates_and_metrics: CandidatesToMetrics,
    ) -> list[list[dict[str, JsonEncodable]]]:
        return [
            [json_conv.unstructure(candidate), json_conv.unstructure(metrics)]
            for candidate, metrics in candidates_and_metrics.items()
        ]

    json_conv.register_unstructure_hook(
        Entity,
        cattrs.gen.make_dict_unstructure_fn(
            Entity,
            json_conv,
            # omit the `_id` if we're not testing, as it's an 'internal' field
            # that users won't want to see in serialized output by default,
            # and it will add to the output size.
            _id=cattrs.gen.override(omit=not testing),
            linking_candidates=cattrs.gen.override(unstruct_hook=_linking_candidates_unstruct_hook),
            _cattrs_include_init_false=True,
            _cattrs_omit_if_default=True,
        ),
    )

    def _linking_candidates_struct_hook(
        candidates_and_metrics: list[list[dict[str, JsonEncodable]]],
        _: type,
    ) -> CandidatesToMetrics:
        return {
            json_conv.structure(candidate_and_metric[0], LinkingCandidate): json_conv.structure(
                candidate_and_metric[1], LinkingMetrics
            )
            for candidate_and_metric in candidates_and_metrics
        }

    json_conv.register_structure_hook(
        Entity,
        cattrs.gen.make_dict_structure_fn(
            Entity,
            json_conv,
            linking_candidates=cattrs.gen.override(struct_hook=_linking_candidates_struct_hook),
            _cattrs_include_init_false=True,
        ),
    )

    json_conv.register_unstructure_hook(
        Section,
        cattrs.gen.make_dict_unstructure_fn(
            Section,
            json_conv,
            _sentence_spans=cattrs.gen.override(rename="sentence_spans", omit_if_default=True),
            _cattrs_include_init_false=True,
            _cattrs_omit_if_default=True,
        ),
    )

    json_conv.register_structure_hook(
        Section,
        cattrs.gen.make_dict_structure_fn(
            Section,
            json_conv,
            _sentence_spans=cattrs.gen.override(
                rename="sentence_spans",
                # needed because cattrs by default ignores init=False fields
                omit=False,
                struct_hook=lambda v, _: (
                    None if len(v) == 0 else tuple(json_conv.structure(x, CharSpan) for x in v)
                ),
            ),
            _cattrs_include_init_false=True,
        ),
    )
    return json_conv


kazu_json_converter = _initialize_json_converter(testing=False)
"""A `cattrs Converter <https://catt.rs/en/stable/converters.html>`_
configured for converting Kazu's datamodel into json.

If you are not familiar with ``cattrs``, don't worry: you can just
use methods on the kazu classes like :meth:`Document.from_dict` and
:meth:`Document.from_dict`, and you will likely never need to use or understand
``kazu_json_converter``.

If you are familiar with cattrs, you may prefer to use the ``structure``, ``unstructure``,
``dumps`` and ``loads`` methods of ``kazu_json_converter`` directly.
"""


class OntologyStringBehaviour(AutoNameEnum):
    #: use the resource for both dictionary based NER and as a linking target.
    ADD_FOR_NER_AND_LINKING = auto()
    #: use the resource only as a linking target. Note, this is not required if the resource is already in the
    #: underlying ontology, as all ontology resources are included as linking targets by default
    #: (Also see DROP_FOR_LINKING)
    ADD_FOR_LINKING_ONLY = auto()
    #: do not use this resource as a linking target. Normally, you would use this for a resource you want to remove
    #: from the underlying ontology (e.g. a 'bad' synonym). If the resource does not exist, has no effect
    DROP_FOR_LINKING = auto()


class ParserBehaviour(AutoNameEnum):
    #: completely remove the ids from the parser - i.e. should never used anywhere
    DROP_IDS_FROM_PARSER = auto()


@dataclass(frozen=True)
class ParserAction:
    """A ParserAction changes the behaviour of a
    :class:`kazu.ontology_preprocessing.base.OntologyParser` in a global sense.

    A ParserAction overrides any default behaviour of the parser, and also any conflicts that may occur with
    :class:`.OntologyStringResource`\\s.

    These actions are useful for eliminating unwanted behaviour. For example, the root of the Mondo
    ontology is http://purl.obolibrary.org/obo/HP_0000001, which has a default label of 'All'. Since this is
    such a common word, and not very useful in terms of linking, we might want a global action so that this
    ID is not used anywhere in a Kazu pipeline.

    The parser_to_target_id_mappings field should specify the parser name and an affected IDs if required.
    See :class:`.ParserBehaviour` for the type of actions that are possible.
    """

    behaviour: ParserBehaviour
    parser_to_target_id_mappings: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, json_dict: dict) -> "ParserAction":
        return kazu_json_converter.structure(json_dict, ParserAction)

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
    """Container for all :class:`.ParserAction`\\s."""

    actions: list[ParserAction]
    _parser_name_to_action: defaultdict[str, list[ParserAction]] = field(
        default_factory=lambda: defaultdict(list), init=False
    )

    def __post_init__(self):
        for action in self.actions:
            for parser_name in action.parser_to_target_id_mappings.keys():
                self._parser_name_to_action[parser_name].append(action)

    def parser_behaviour(self, parser_name: str) -> Iterable[ParserAction]:
        """Generator that yields behaviours for a specific parser, based on the order
        they are specified in.

        :param parser_name:
        :return:
        """
        yield from self._parser_name_to_action.get(parser_name, [])

    @classmethod
    def from_dict(cls, json_dict: dict) -> "GlobalParserActions":
        return kazu_json_converter.structure(json_dict, GlobalParserActions)


@dataclass(frozen=True)
class Synonym:
    text: str
    case_sensitive: bool
    mention_confidence: MentionConfidence


@dataclass(frozen=True)
class OntologyStringResource:
    """A OntologyStringResource represents the behaviour of a specific
    :class:`.LinkingCandidate` within an Ontology.

    For each LinkingCandidate, a default OntologyStringResource is produced with its
    behaviour determined by an instance of
    :class:`kazu.ontology_preprocessing.autocuration.AutoCurator` and the
    :class:`kazu.ontology_preprocessing.curation_utils.OntologyStringConflictAnalyser`\\.

    .. note::

       This is typically handled by the internals of :class:`kazu.ontology_preprocessing.base.OntologyParser`\\.
       However, OntologyStringResources can also be used to override the default behaviour of a parser. See :ref:`ontology_parser`
       for a more detailed guide.

    The configuration of a OntologyStringResource will affect both NER and Linking aspects of Kazu:

    Example 1:

    The string 'ALL' is highly ambiguous. It might mean several diseases, or simply 'all'. Therefore, we want
    to add a curation as follows, so that it will only be used as a linking target and not for dictionary based NER:

    .. code-block:: python

        OntologyStringResource(
            original_synonyms=frozenset(
                [
                    Synonym(
                        text="ALL",
                        mention_confidence=MentionConfidence.POSSIBLE,
                        case_sensitive=True,
                    )
                ]
            ),
            behaviour=OntologyStringBehaviour.ADD_FOR_LINKING_ONLY,
        )


    Example 2:

    The string 'LH' is incorrectly identified as a synonym of the PLOD1 (ENSG00000083444) gene, whereas more often than not, it's actually an abbreviation of Lutenising Hormone.
    We therefore want to override the associated_id_sets to LHB (ENSG00000104826, or Lutenising Hormone Subunit Beta)

    The OntologyStringResource we therefore want is:

    .. code-block:: python

        OntologyStringResource(
            original_synonyms=frozenset(
                [
                    Synonym(
                        text="LH",
                        mention_confidence=MentionConfidence.POSSIBLE,
                        case_sensitive=True,
                    )
                ]
            ),
            associated_id_sets=frozenset((EquivalentIdSet(("ENSG00000104826", "ENSEMBL")),)),
            behaviour=OntologyStringBehaviour.ADD_FOR_LINKING_ONLY,
        )

    Example 3:

    A :class:`.LinkingCandidate` has an alternative synonym not referenced in the underlying ontology, and we want to add it.

    .. code-block:: python

        OntologyStringResource(
            original_synonyms=frozenset(
                [
                    Synonym(
                        text="breast carcinoma",
                        mention_confidence=MentionConfidence.POSSIBLE,
                        case_sensitive=True,
                    )
                ]
            ),
            associated_id_sets=frozenset((EquivalentIdSet(("ENSG00000104826", "ENSEMBL")),)),
            behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
        )
    """

    #: Original synonyms, exactly as specified in the source ontology. These should all normalise to the same string.
    original_synonyms: frozenset[Synonym]
    #: The intended behaviour for this resource.
    behaviour: OntologyStringBehaviour
    #: Alternative synonyms generated from the originals by :class:`kazu.ontology_preprocessing.synonym_generation.CombinatorialSynonymGenerator`\.
    alternative_synonyms: frozenset[Synonym] = field(default_factory=frozenset)
    #: If specified, will override the parser defaults for the associated :class:`.LinkingCandidate`\, as long as conflicts do not occur
    associated_id_sets: Optional[AssociatedIdSets] = None
    _id: bson.ObjectId = field(default_factory=bson.ObjectId, compare=False)
    #: results of any decisions by the :class:`kazu.ontology_preprocessing.autocuration.AutoCurator`
    autocuration_results: Optional[dict[str, str]] = field(default=None, compare=False)
    #: human readable comments about this curation decision
    comment: Optional[str] = field(default=None, compare=False)

    def __post_init__(self):
        cs_syns = defaultdict(set)
        ci_syns = defaultdict(set)
        if len(self.original_synonyms) == 0:
            raise ValueError(f"no synonyms for: {self}")
        for syn in self.active_ner_synonyms():
            if syn.case_sensitive:
                cs_syns[syn.text].add(syn.mention_confidence)
            else:
                ci_syns[syn.text.lower()].add(syn.mention_confidence)
        for cs_syn, cs_confidences in ci_syns.items():
            ci_confidences = ci_syns.get(cs_syn.lower(), {MentionConfidence.POSSIBLE})
            if min(ci_confidences) < min(cs_confidences):
                raise ValueError(f"case sensitive conflict: {self}")

    def syn_norm_for_linking(self, entity_class: str) -> str:
        norms = set(
            StringNormalizer.normalize(syn.text, entity_class) for syn in self.original_synonyms
        )
        if len(norms) == 1:
            syn_norm = next(iter(norms))
            return syn_norm
        else:
            raise ValueError(
                f"multiple synonym norms produced by {self}. This resource should be separated into two or more separate items."
            )

    @staticmethod
    def from_json(json_str: str) -> "OntologyStringResource":
        json_dict = json.loads(json_str)
        return OntologyStringResource.from_dict(json_dict)

    @classmethod
    def from_dict(cls, json_dict: dict) -> "OntologyStringResource":
        return kazu_json_converter.structure(json_dict, OntologyStringResource)

    def to_dict(self, preserve_structured_object_id: bool = True) -> dict[str, Any]:
        as_dict: dict[str, Any] = kazu_json_converter.unstructure(self)
        if preserve_structured_object_id:
            as_dict["_id"] = self._id
        return as_dict

    def to_json(self) -> str:
        as_json = self.to_dict(False)
        assert isinstance(as_json, dict)
        return json.dumps(as_json)

    @property
    def additional_to_source(self) -> bool:
        """True if this resource created in addition to the source resources defined in
        the original Ontology."""
        return self.associated_id_sets is not None and self.behaviour in {
            OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
            OntologyStringBehaviour.ADD_FOR_LINKING_ONLY,
        }

    def all_synonyms(self) -> Iterable[Synonym]:
        for syn in self.original_synonyms.union(self.alternative_synonyms):
            yield syn

    def all_strings(self) -> Iterable[str]:
        for syn in self.all_synonyms():
            yield syn.text

    def active_ner_synonyms(self) -> Iterable[Synonym]:
        if self.behaviour is OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING:
            for syn in self.original_synonyms.union(self.alternative_synonyms):
                if syn.mention_confidence is not MentionConfidence.IGNORE:
                    yield syn


class KazuConfigurationError(Exception):
    pass
