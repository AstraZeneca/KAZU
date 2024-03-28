import logging
import pickle
import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path
from typing import Any, Union, Optional, cast

import spacy
import srsly
from spacy.language import Language
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span, SpanGroup, Doc, Token
from spacy.util import SimpleFrozenList

from kazu.ontology_preprocessing.base import (
    OntologyParser,
)
from kazu.utils.grouping import sort_then_group
from kazu.utils.utils import PathLike

GENE = "gene"
DRUG = "drug"
ANATOMY = "anatomy"
DISEASE = "disease"
CELL_LINE = "cell_line"
CELL_TYPE = "cell_type"

SPAN_KEY = "RAW_HITS"
MATCH_ID_SEP = ":::"

logger = logging.getLogger(__name__)

_CoocDict = dict[str, dict[str, list[str]]]
_MatcherDict = dict[str, Matcher]


@dataclass
class OntologyMatcherConfig:
    span_key: str
    match_id_sep: str
    labels: list[str]
    parser_name_to_entity_type: dict[str, str]


# the strings in the tuple are: parser_name, syn_norm, confidence value
_MatcherOntologyData = dict[str, set[tuple[str, str, str]]]


def _ontology_dict_setter(span: Span, value: dict) -> None:
    # spacy's typing on the property says this has to be a dict[str, Any]
    # but the __init__ typing says it can be a dict[Any, Any]
    span.doc.user_data[span] = value  # type: ignore[index]


def _ontology_dict_getter(span: Span) -> _MatcherOntologyData:
    # as above with types
    return cast(_MatcherOntologyData, span.doc.user_data.get(span, {}))  # type: ignore[call-overload]


@Language.factory(
    "ontology_matcher",
    default_config={
        "span_key": SPAN_KEY,
        "match_id_sep": MATCH_ID_SEP,
    },
)
class OntologyMatcher:
    """String matching to synonyms.

    Core strict matching is done by spaCy's
    `PhraseMatcher <https://spacy.io/api/phrasematcher>`_.
    """

    def __init__(
        self,
        nlp: Language,
        name: str = "ontology_matcher",
        *,
        span_key: str = SPAN_KEY,
        match_id_sep: str = MATCH_ID_SEP,
        parser_name_to_entity_type: dict[str, str],
    ):
        """
        :param nlp: a spacy model, used for its vocab and tokenizer
        :param name: the name of this component. Used for spacy config
        :param span_key: the key for doc.spans to store the matches in
        :param match_id_sep: a separator this splits fields in the match id
        :param parser_name_to_entity_type: a mapping from parsers to their entity class
        """

        Span.set_extension(
            "ontology_dict_", force=True, getter=_ontology_dict_getter, setter=_ontology_dict_setter
        )
        self.nlp = nlp
        self.name = name
        self.cfg = OntologyMatcherConfig(
            span_key=span_key,
            match_id_sep=match_id_sep,
            labels=[],
            parser_name_to_entity_type=parser_name_to_entity_type,
        )
        # These will be defined when calling initialize
        self.strict_matcher: Optional[PhraseMatcher] = None
        self.lowercase_matcher: Optional[PhraseMatcher] = None
        self.tp_matchers: Optional[_MatcherDict] = None
        self.fp_matchers: Optional[_MatcherDict] = None
        self.tp_coocc_dict: Optional[_CoocDict] = None
        self.fp_coocc_dict: Optional[_CoocDict] = None

    @property
    def nr_strict_rules(self) -> int:
        if not self.strict_matcher:
            return 0
        else:
            return len(self.strict_matcher)

    @property
    def nr_lowercase_rules(self) -> int:
        if not self.lowercase_matcher:
            return 0
        else:
            return len(self.lowercase_matcher)

    @property
    def span_key(self) -> str:
        return self.cfg.span_key

    @property
    def match_id_sep(self) -> str:
        return self.cfg.match_id_sep

    @property
    def labels(self) -> list[str]:
        """The labels currently processed by this component."""
        return self.cfg.labels

    @property
    def parser_name_to_entity_type(self) -> dict[str, str]:
        return self.cfg.parser_name_to_entity_type

    def set_labels(self, labels: Iterable[str]) -> None:
        self.cfg.labels = list(labels)
        self.set_context_matchers()

    def set_context_matchers(self):
        self.tp_matchers, self.fp_matchers = self._create_token_matchers()
        self.tp_coocc_dict, self.fp_coocc_dict = self._create_coocc_dicts()

    def create_phrasematchers(
        self, parsers: list[OntologyParser]
    ) -> tuple[Optional[PhraseMatcher], Optional[PhraseMatcher]]:
        """Create spaCy `PhraseMatcher <https://spacy.io/api/phrasematcher>`_\\s.

        :class:`.OntologyStringResource`\\ s are produced by :py:meth:`.OntologyParser.populate_databases`\\ method.

        :param parsers:
        :return:
        """
        if self.strict_matcher is not None or self.lowercase_matcher is not None:
            logging.warning("Phrase matchers are being redefined - is this by intention?")

        self.set_labels(parser.entity_class for parser in parsers)
        strict_matcher = PhraseMatcher(self.nlp.vocab, attr="ORTH")
        lowercase_matcher = PhraseMatcher(self.nlp.vocab, attr="NORM")
        logger.info("ontology matcher build triggered.")
        for parser in parsers:
            parser_resources = parser.populate_databases(return_resources=True)
            if parser_resources is None:
                logger.warning(
                    "tried to create PhraseMatchers from Resources for parser %s, but none have been provided",
                    parser.name,
                )
                continue
            if len(parser_resources) == 0:
                logger.warning(
                    "tried to create PhraseMatchers from Resources for parser %s, but no Resources were produced",
                    parser.name,
                )
                continue

            # deduplicating match id's and patterns saves memory in the spacy pipeline
            match_ids_and_text_cs = set()
            match_ids_and_text_ci = set()
            for resource in parser_resources:
                # a resource can have different syn_norms for different parsers,
                # since the string normalizer's output depends on the entity class.
                # Also, a synonym may exist in multiple LinkingCandidate.raw_synonyms
                syn_norm = resource.syn_norm_for_linking(parser.entity_class)
                for syn in resource.active_ner_synonyms():
                    match_id = (
                        parser.name
                        + self.match_id_sep
                        + syn_norm
                        + self.match_id_sep
                        + str(syn.mention_confidence.value)
                    )
                    if syn.case_sensitive:
                        match_ids_and_text_cs.add(
                            (
                                match_id,
                                syn.text,
                            )
                        )
                    else:
                        match_ids_and_text_ci.add(
                            (
                                match_id,
                                syn.text.lower(),
                            )
                        )

            for match_id, match_str in match_ids_and_text_cs:
                strict_matcher.add(match_id, [self.nlp.tokenizer(match_str)])
            for match_id, match_str in match_ids_and_text_ci:
                lowercase_matcher.add(match_id, [self.nlp.tokenizer(match_str)])

        # only set the phrasematcher if we have any rules for them
        # this lets us skip running a phrasematcher if it has no rules when we come
        # to calling OntologyMatcher
        self.strict_matcher = strict_matcher if len(strict_matcher) != 0 else None
        self.lowercase_matcher = lowercase_matcher if len(lowercase_matcher) != 0 else None
        return self.strict_matcher, self.lowercase_matcher

    def __call__(self, doc: Doc) -> Doc:
        if self.nr_strict_rules == 0 and self.nr_lowercase_rules == 0:
            raise ValueError("there are no matcher rules configured!")

        if self.strict_matcher is not None and self.lowercase_matcher is not None:
            matches = set(self.strict_matcher(doc)).union(set(self.lowercase_matcher(doc)))
        elif self.strict_matcher is not None:
            matches = set(self.strict_matcher(doc))
        elif self.lowercase_matcher is not None:
            matches = set(self.lowercase_matcher(doc))
        else:
            # this isn't possible since if both of them were None,
            # the if clause at the start of this function would raise an error
            raise AssertionError()

        spans: list[Span] = []
        for (start, end), matches_grp in sort_then_group(
            matches,
            key_func=lambda x: (
                x[1],
                x[2],
            ),
        ):
            data = defaultdict(set)
            for mat in matches_grp:
                parser_name, syn_norm, confidence = self.nlp.vocab.strings.as_string(mat[0]).split(
                    self.match_id_sep, maxsplit=2
                )
                ent_class = self.parser_name_to_entity_type[parser_name]
                data[ent_class].add((parser_name, syn_norm, confidence))
            for ent_class, ent_class_data in data.items():
                # we use a uuid here so that every span hash is unique
                new_span = Span(doc, start, end, label=uuid.uuid4().hex)
                new_span._.set("ontology_dict_", {ent_class: ent_class_data})
                spans.append(new_span)
        final_spans = self.filter_by_contexts(doc, spans)
        span_group = SpanGroup(doc, name=self.span_key, spans=final_spans)
        doc.spans[self.span_key] = span_group
        return doc

    def filter_by_contexts(self, doc: Doc, spans: list[Span]) -> list[Span]:
        """These filters work best when there is sentence segmentation available."""
        doc_has_sents = doc.has_annotation("SENT_START")
        # set custom attributes for the token matchers
        for label in self.labels:
            Token.set_extension(label, default=False, force=True)
        # compile the filtered spans by going through each label and each span
        filtered_spans: list[Span] = []
        for label in self.labels:
            # Set all token attributes for other labels (to use in the matcher rules)
            for s in spans:
                for ent_class in s._.ontology_dict_:
                    if ent_class != label and ent_class in self.labels:
                        self._set_token_attr(ent_class=ent_class, span=s, value=True)
            # Set the token attribute for each span of this label one-by-one, then match
            for span in spans:
                for ent_class in span._.ontology_dict_:
                    if ent_class == label:
                        self._set_token_attr(ent_class=ent_class, span=span, value=True)
                        context: Union[Doc, Span] = doc
                        if doc_has_sents:
                            context = span.sent
                        if (
                            self.span_in_TP_coocc(context, span=span, ent_class=ent_class)
                            and not self.span_in_FP_coocc(context, span=span, ent_class=ent_class)
                            and self.span_in_TP_context(context, ent_class)
                            and not self.span_in_FP_context(context, ent_class)
                        ):
                            filtered_spans.append(span)
                        # reset for the next span within the same label
                        self._set_token_attr(ent_class, span, False)
            # reset for the next label
            for s in spans:
                for ent_class in s._.ontology_dict_:
                    if ent_class in self.labels:
                        self._set_token_attr(ent_class, s, False)
        return filtered_spans

    def _set_token_attr(self, ent_class: str, span: Span, value: bool) -> None:
        for token in span:
            token._.set(ent_class, value)

    def span_in_TP_context(self, doc: Union[Doc, Span], ent_class: str) -> bool:
        """When an entity type has a TP matcher defined, it should match for this span
        to be regarded as a true hit."""
        assert self.tp_matchers is not None
        tp_matcher = self.tp_matchers.get(ent_class, None)
        if tp_matcher:
            rule_spans = tp_matcher(doc, as_spans=True)
            if len(rule_spans) > 0:
                return True
            return False
        return True

    def span_in_FP_context(self, doc: Union[Doc, Span], ent_class: str) -> bool:
        """When an entity type has a FP matcher defined, spans that match are regarded
        as FPs."""
        assert self.fp_matchers is not None
        fp_matcher = self.fp_matchers.get(ent_class, None)
        if fp_matcher:
            rule_spans = fp_matcher(doc, as_spans=True)
            if len(rule_spans) > 0:
                return True
            return False
        return False

    def span_in_TP_coocc(self, doc: Union[Doc, Span], span: Span, ent_class: str) -> bool:
        """When an entity type has a TP co-occ dict defined, a hit defined in the dict
        is only regarded as a true hit when it matches at least one of its co-occ
        terms."""
        assert self.tp_coocc_dict is not None
        tp_dict = self.tp_coocc_dict.get(ent_class, None)
        if tp_dict and tp_dict.get(span.text):
            for w in tp_dict[span.text]:
                if w in doc.text:
                    return True
            return False
        return True

    def span_in_FP_coocc(self, doc: Union[Doc, Span], span: Span, ent_class: str) -> bool:
        """When an entity type has a FP co-occ dic defined, a hit defined in the dict is
        regarded as a false positive when it matches at least one of its co-occ
        terms."""
        assert self.fp_coocc_dict is not None
        fp_dict = self.fp_coocc_dict.get(ent_class, None)
        if fp_dict and fp_dict.get(span.text):
            for w in fp_dict[span.text]:
                if w in doc.text:
                    return True
            return False
        return False

    def _create_token_matchers(self) -> tuple[_MatcherDict, _MatcherDict]:
        tp_matchers: _MatcherDict = {}
        fp_matchers: _MatcherDict = {}
        if CELL_LINE in self.labels:
            tp_matchers[CELL_LINE] = self._create_cell_tp_tokenmatcher(CELL_LINE)
        if CELL_TYPE in self.labels:
            tp_matchers[CELL_TYPE] = self._create_cell_tp_tokenmatcher(CELL_TYPE)

        if ANATOMY in self.labels:
            fp_matchers[ANATOMY] = self._create_anatomy_fp_tokenmatcher()
        return tp_matchers, fp_matchers

    def _create_cell_tp_tokenmatcher(self, ent_class: str) -> Matcher:
        """Define patterns where a Cell line or type appears and it's likely a true
        positive."""
        matcher = Matcher(self.nlp.vocab)
        pattern_1: list[dict[str, Any]] = [
            {"_": {ent_class: True}},
            {"LOWER": {"IN": ["cell", "cells"]}},
        ]
        pattern_2: list[dict[str, Any]] = [
            {"LOWER": "cell"},
            {"LOWER": "line"},
            {"_": {ent_class: True}},
        ]
        pattern_3: list[dict[str, Any]] = [
            {"LOWER": "cell"},
            {"LOWER": "type"},
            {"_": {ent_class: True}},
        ]
        matcher.add("Cell_context", [pattern_1, pattern_2, pattern_3])
        return matcher

    def _create_anatomy_fp_tokenmatcher(self) -> Matcher:
        """Define patterns where an atanomy entity appears and it's likely a false
        positive."""
        matcher = Matcher(self.nlp.vocab)
        patterns: list[list[dict[str, Any]]] = []
        if DRUG in self.labels:
            p: list[dict[str, Any]] = [{"_": {DRUG: True}}, {"_": {ANATOMY: True}, "LOWER": "arm"}]
            patterns.append(p)
        p2: list[dict[str, Any]] = [
            {"LOWER": "single"},
            {"LOWER": "-"},
            {"_": {ANATOMY: True}, "LOWER": "arm"},
        ]
        patterns.append(p2)
        p3: list[dict[str, Any]] = [
            {"LOWER": "quality"},
            {"LOWER": "-", "OP": "?"},
            {"LOWER": "of"},
            {"LOWER": "-", "OP": "?"},
            {"_": {ANATOMY: True}, "LOWER": "life"},
        ]
        patterns.append(p3)
        matcher.add("Anatomy_context", patterns)
        return matcher

    def _create_coocc_dicts(self):
        tp_coocc_dict: _CoocDict = {}
        fp_coocc_dict: _CoocDict = {}
        if GENE in self.labels:
            fp_coocc_dict[GENE] = self._create_gene_fp_dict()
        if DISEASE in self.labels:
            fp_coocc_dict[DISEASE] = self._create_disease_fp_dict()
        return tp_coocc_dict, fp_coocc_dict

    _ivf_fertility_treatment_cooccurrence = ["ICSI", "cycle", "treatment"]

    def _create_gene_fp_dict(self):
        """Define cooccurrence links that determine likely FP gene hits."""
        gene_dict = {}
        gene_dict["IVF"] = self._ivf_fertility_treatment_cooccurrence
        return gene_dict

    def _create_disease_fp_dict(self):
        """Define cooccurrence links that determine likely FP disease hits."""
        disease_dict = {}
        disease_dict["MFS"] = ["endpoint"]
        disease_dict["IVF"] = self._ivf_fertility_treatment_cooccurrence
        return disease_dict

    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> None:
        """Serialize the pipe to disk.

        :param path: Path to serialize the pipeline to.
        :param exclude: String names of serialization fields to exclude.
        """

        def pickle_matcher(p, strict):
            with p.open("wb") as outfile:
                if strict:
                    pickle.dump(self.strict_matcher, outfile)
                else:
                    pickle.dump(self.lowercase_matcher, outfile)

        serialize = {}
        serialize["cfg"] = lambda p: srsly.write_json(p, asdict(self.cfg))
        serialize["strict_matcher"] = partial(pickle_matcher, strict=True)
        serialize["lowercase_matcher"] = partial(pickle_matcher, strict=False)
        spacy.util.to_disk(path, serialize, exclude)

    def from_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> "OntologyMatcher":
        """Load the pipe from disk.

        Modifies the object in place and returns it.
        """

        def unpickle_matcher(p, strict):
            with p.open("rb") as infile:
                matcher = pickle.load(infile)
                if strict:
                    self.strict_matcher = matcher
                else:
                    self.lowercase_matcher = matcher

        deserialize = {}

        def deserialize_cfg(path: PathLike) -> None:
            loaded_conf = srsly.read_json(path)
            self.cfg = OntologyMatcherConfig(**loaded_conf)

        deserialize["cfg"] = deserialize_cfg
        deserialize["strict_matcher"] = partial(unpickle_matcher, strict=True)
        deserialize["lowercase_matcher"] = partial(unpickle_matcher, strict=False)
        # spacy's typing says the functions have to only take strings, but they would normally take
        # a path as well without that being a problem.
        spacy.util.from_disk(path, deserialize, exclude)  # type: ignore[arg-type]
        self.set_context_matchers()
        return self
