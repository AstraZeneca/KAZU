import json
import logging
import pickle
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path
from typing import List, Dict, Union, Callable, Iterable, Tuple

import spacy
import srsly

from kazu.modelling.ontology_preprocessing.base import OntologyParser
from kazu.utils.utils import PathLike
from spacy import Language
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span, SpanGroup, Doc, Token
from spacy.util import SimpleFrozenList

GENE = "gene"
DRUG = "drug"
ANATOMY = "anatomy"
DISEASE = "disease"
CELL_LINE = "cell_line"
ENTITY = "entity"

SPAN_KEY = "RAW_HITS"
MATCH_ID_SEP = ":::"


@dataclass
class OntologyMatcherConfig:
    span_key: str
    match_id_sep: str
    labels: List[str]
    parser_name_to_entity_type: Dict[str, str]


@Language.factory(
    "ontology_matcher",
    default_config={
        "span_key": SPAN_KEY,
        "match_id_sep": MATCH_ID_SEP,
    },
)
class OntologyMatcher:
    """String matching to synonyms.

    Core strict matching is done by Spacy's `PhraseMatcher <https://spacy.io/api/phrasematcher>`_.
    """

    def __init__(
        self,
        nlp: Language,
        name: str = "ontology_matcher",
        *,
        span_key: str = SPAN_KEY,
        match_id_sep: str = MATCH_ID_SEP,
        entry_filter: Callable[[str, str], Tuple[bool, bool]],
        parser_name_to_entity_type: Dict[str, str],
    ):
        """

        :param span_key: the key for doc.spans to store the matches in
        :param entry_filter: a function deciding whether a given ontology row/entry is valid
        """
        self.nlp = nlp
        self.name = name
        self.entry_filter = entry_filter
        self.cfg = OntologyMatcherConfig(
            span_key=span_key,
            match_id_sep=match_id_sep,
            labels=[],
            parser_name_to_entity_type=parser_name_to_entity_type,
        )
        # These will be defined when calling initialize
        self.strict_matcher, self.lowercase_matcher = None, None
        self.tp_matchers, self.fp_matchers = None, None
        self.tp_coocc_dict, self.fp_coocc_dict = None, None

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
    def labels(self) -> List[str]:
        """The labels currently processed by this component."""
        return self.cfg.labels

    @property
    def parser_name_to_entity_type(self) -> Dict[str, str]:
        return self.cfg.parser_name_to_entity_type

    def set_labels(self, labels: Iterable[str]):
        self.cfg.labels = list(labels)
        self.set_context_matchers()

    def set_context_matchers(self):
        self.tp_matchers, self.fp_matchers = self._create_token_matchers()
        self.tp_coocc_dict, self.fp_coocc_dict = self._create_coocc_dicts()

    def create_phrasematchers_from_curated_list(
        self, curated_list: PathLike
    ) -> Tuple[PhraseMatcher, PhraseMatcher]:
        if self.strict_matcher is not None or self.lowercase_matcher is not None:
            logging.warning("Phrase matchers are being redefined - is this by intention?")

        strict_matcher = PhraseMatcher(self.nlp.vocab, attr="ORTH")
        lowercase_matcher = PhraseMatcher(self.nlp.vocab, attr="NORM")

        with open(curated_list, mode="r") as jsonlf:
            curated_synonyms = [json.loads(line) for line in jsonlf]

        synonyms_to_add = (item for item in curated_synonyms if item["action"] == "keep")

        patterns = self.nlp.tokenizer.pipe(
            # we need it lowercased for the case insensitive matcher
            item["term"] if item["case_sensitive"] else item["term"].lower()
            for item in synonyms_to_add
        )

        for item, pattern in zip(curated_synonyms, patterns):
            # a generated synonym can have different term_norms for different parsers,
            # since the string normalizer's output depends on the entity class
            for parser_name, term_norm in item["term_norm_mapping"].items():
                match_id = parser_name + self.match_id_sep + term_norm
                if item["case_sensitive"]:
                    strict_matcher.add(match_id, [pattern])
                else:
                    lowercase_matcher.add(match_id, [pattern])

        # only set the phrasematcher if we have any rules for them
        # this lets us skip running a phrasematcher if it has no rules when we come
        # to calling OntologyMatcher
        self.strict_matcher = strict_matcher if len(strict_matcher) != 0 else None
        self.lowercase_matcher = lowercase_matcher if len(lowercase_matcher) != 0 else None
        return strict_matcher, lowercase_matcher

    def create_lowercase_phrasematcher_from_parsers(
        self, parsers: List[OntologyParser]
    ) -> Tuple[PhraseMatcher, None]:
        """Initialize the phrase matchers when creating this component.
        This method should not be run on an existing or deserialized pipeline.

        Returns the lowercase phrasematcher and None - so it matches the return shape of
        :meth:`create_phrasematchers_from_curated_list`\\ .
        """
        if self.strict_matcher is not None or self.lowercase_matcher is not None:
            logging.warning("Phrase matchers are being redefined - is this by intention?")

        lowercase_matcher = PhraseMatcher(self.nlp.vocab, attr="NORM")
        for parser in parsers:
            synonym_terms = parser.generate_synonyms()
            parser_name = parser.name
            logging.info(
                f"generating {sum(len(x.terms) for x in synonym_terms)} patterns for {parser_name}"
            )
            patterns = self.nlp.tokenizer.pipe(
                term for synonym in synonym_terms for term in synonym.terms
            )
            for synonym_term, pattern in zip(synonym_terms, patterns):
                match_id = parser_name + self.match_id_sep + synonym_term.term_norm
                for syn in synonym_term.terms:
                    try:
                        # if we're adding to the lowercase matcher, we don't need to add
                        # to the exact case matcher as well, since we'll definitely get
                        # the hit, so would just be a waste of memory and compute.
                        lowercase_matcher.add(match_id, [pattern])

                    except KeyError as e:
                        logging.warning(
                            f"failed to add '{syn}'. StringStore is {len(self.nlp.vocab.strings)} ",
                            e,
                        )

        self.lowercase_matcher = lowercase_matcher
        return lowercase_matcher, None

    def __call__(self, doc: Doc) -> Doc:
        if self.nr_strict_rules == 0 and self.nr_lowercase_rules == 0:
            raise ValueError(
                "The matcher rules have not been set up properly. "
                "Did you initialize the labels and the phrase matchers?"
            )

        # implied by above - strict_matcher is None implies self.nr_strict_rules
        # and equivalent for lowercase matches
        # mypy is happier having this assertion
        assert self.strict_matcher is not None or self.lowercase_matcher is not None

        # at least one phrasematcher will now be set.
        # normally, this will only be one: either a strict matcher if constructed by curated list,
        # or a lowercase matcher if constructed 'from scrach' using the parsers - currently just an
        # initial step in building a curation-based phrasematcher

        if self.strict_matcher is not None and self.lowercase_matcher is not None:
            matches = set(self.strict_matcher(doc)).union(set(self.lowercase_matcher(doc)))
        elif self.strict_matcher is not None:
            matches = set(self.strict_matcher(doc))
        elif self.lowercase_matcher is not None:
            matches = set(self.lowercase_matcher(doc))

        spans = set(Span(doc, start, end, label=key) for key, start, end in matches)
        spans = self._set_span_attributes(spans)
        final_spans = self.filter_by_contexts(doc, spans)
        self.set_annotations(doc, final_spans)
        return doc

    def set_annotations(self, doc, spans):
        span_key = self.span_key
        span_group = SpanGroup(doc, name=span_key, spans=spans)
        doc.spans[span_key] = span_group

    def filter_by_contexts(self, doc, spans):
        """These filters work best when there is sentence segmentation available."""
        doc_has_sents = doc.has_annotation("SENT_START")
        # set custom attributes for the token matchers
        for label in self.labels:
            Token.set_extension(label, default=False, force=True)
        # compile the filtered spans by going through each label and each span
        filtered_spans = []
        for label in self.labels:
            # Set all token attributes for other labels (to use in the matcher rules)
            for s in spans:
                if s.label_ != label and s.label_ in self.labels:
                    self._set_token_attr(s, True)
            # Set the token attribute for each span of this label one-by-one, then match
            for span in spans:
                if span.label_ == label:
                    self._set_token_attr(span, True)
                    context = doc
                    if doc_has_sents:
                        context = span.sent
                    if (
                        self.span_in_TP_coocc(context, span)
                        and not self.span_in_FP_coocc(context, span)
                        and self.span_in_TP_context(context, span)
                        and not self.span_in_FP_context(context, span)
                    ):
                        filtered_spans.append(span)
                    # reset for the next span within the same label
                    self._set_token_attr(span, False)
            # reset for the next label
            for s in spans:
                if s.label_ in self.labels:
                    self._set_token_attr(s, False)
        return filtered_spans

    def _set_token_attr(self, span, value: bool):
        for token in span:
            token._.set(span.label_, value)

    def span_in_TP_context(self, doc, span):
        """When an entity type has a TP matcher defined, it should match for this
        span to be regarded as a true hit."""
        tp_matcher = self.tp_matchers.get(span.label_, None)
        if tp_matcher:
            rule_spans = tp_matcher(doc, as_spans=True)
            if len(rule_spans) > 0:
                return True
            return False
        return True

    def span_in_FP_context(self, doc, span):
        """When an entity type has a FP matcher defined, spans that match
        are regarded as FPs."""
        fp_matcher = self.fp_matchers.get(span.label_, None)
        if fp_matcher:
            rule_spans = fp_matcher(doc, as_spans=True)
            if len(rule_spans) > 0:
                return True
            return False
        return False

    def span_in_TP_coocc(self, doc, span):
        """When an entity type has a TP co-occ dict defined, a hit defined in the dict
        is only regarded as a true hit when it matches at least one of its co-occ terms."""
        tp_dict = self.tp_coocc_dict.get(span.label_, None)
        if tp_dict and tp_dict.get(span.text):
            for w in tp_dict[span.text]:
                if w in doc.text:
                    return True
            return False
        return True

    def span_in_FP_coocc(self, doc, span):
        """When an entity type has a FP co-occ dic defined, a hit defined in the dict
        is regarded as a false positive when it matches at least one of its co-occ terms."""
        fp_dict = self.fp_coocc_dict.get(span.label_, None)
        if fp_dict and fp_dict.get(span.text):
            for w in fp_dict[span.text]:
                if w in doc.text:
                    return True
            return False
        return False

    def _set_span_attributes(self, spans):
        for span in spans:
            span.parser_name_, span.term_norm_ = span.label_.split(self.match_id_sep, maxsplit=1)
            span.label_ = self.parser_name_to_entity_type[span.parser_name_]
        return spans

    def _create_token_matchers(self):
        tp_matchers = {}
        fp_matchers = {}
        if CELL_LINE in self.labels:
            tp_matchers[CELL_LINE] = self._create_cellline_tp_tokenmatcher()

        if ANATOMY in self.labels:
            fp_matchers[ANATOMY] = self._create_anatomy_fp_tokenmatcher()
        return tp_matchers, fp_matchers

    def _create_cellline_tp_tokenmatcher(self):
        """Define patterns where a Cell line appears and it's likely a true positive"""
        matcher = Matcher(self.nlp.vocab)
        pattern_1 = [{"_": {CELL_LINE: True}}, {"LOWER": {"IN": ["cell", "cells"]}}]
        pattern_2 = [{"LOWER": "cell"}, {"LOWER": "line"}, {"_": {CELL_LINE: True}}]
        pattern_3 = [{"LOWER": "cell"}, {"LOWER": "type"}, {"_": {CELL_LINE: True}}]
        matcher.add("Cell_line_context", [pattern_1, pattern_2, pattern_3])
        return matcher

    def _create_anatomy_fp_tokenmatcher(self):
        """Define patterns where an atanomy entity appears and it's likely a false positive"""
        matcher = Matcher(self.nlp.vocab)
        patterns = []
        if DRUG in self.labels:
            p = [{"_": {DRUG: True}}, {"_": {ANATOMY: True}, "LOWER": "arm"}]
            patterns.append(p)
        p2 = [
            {"LOWER": "single"},
            {"LOWER": "-"},
            {"_": {ANATOMY: True}, "LOWER": "arm"},
        ]
        patterns.append(p2)
        p3 = [
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
        tp_coocc_dict: Dict[str, Dict[str, List[str]]] = {}
        fp_coocc_dict: Dict[str, Dict[str, List[str]]] = {}
        if GENE in self.labels:
            fp_coocc_dict[GENE] = self._create_gene_fp_dict()
        if DISEASE in self.labels:
            fp_coocc_dict[DISEASE] = self._create_disease_fp_dict()
        return tp_coocc_dict, fp_coocc_dict

    _ivf_fertility_treatment_cooccurrence = ["ICSI", "cycle", "treatment"]

    def _create_gene_fp_dict(self):
        """Define cooccurrence links that determine likely FP gene hits"""
        gene_dict = {}
        gene_dict["IVF"] = self._ivf_fertility_treatment_cooccurrence
        return gene_dict

    def _create_disease_fp_dict(self):
        """Define cooccurrence links that determine likely FP disease hits"""
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
        """Load the pipe from disk. Modifies the object in place and returns it."""

        def unpickle_matcher(p, strict):
            with p.open("rb") as infile:
                matcher = pickle.load(infile)
                if strict:
                    self.strict_matcher = matcher
                else:
                    self.lowercase_matcher = matcher

        deserialize = {}

        def deserialize_cfg(path: str) -> None:
            loaded_conf = srsly.read_json(path)
            self.cfg = OntologyMatcherConfig(**loaded_conf)

        deserialize["cfg"] = deserialize_cfg
        deserialize["strict_matcher"] = partial(unpickle_matcher, strict=True)
        deserialize["lowercase_matcher"] = partial(unpickle_matcher, strict=False)
        spacy.util.from_disk(path, deserialize, exclude)
        self.set_context_matchers()
        return self
