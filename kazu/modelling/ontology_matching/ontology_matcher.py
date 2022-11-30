import json
import logging
import pickle
import uuid
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from functools import partial
from pathlib import Path
from typing import List, Dict, Union, Iterable, Tuple, Optional, DefaultDict, Set

import spacy
import srsly

from kazu.data.data import SynonymTerm
from kazu.modelling.ontology_preprocessing.base import OntologyParser
from kazu.utils.grouping import sort_then_group
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

logger = logging.getLogger(__name__)


@dataclass
class OntologyMatcherConfig:
    span_key: str
    match_id_sep: str
    labels: List[str]
    parser_name_to_entity_type: Dict[str, str]


@dataclass
class CuratedTerm:
    term: str
    action: str
    case_sensitive: bool
    entity_class: str
    term_norm_mapping: Dict[str, Set[str]] = field(default_factory=dict)


def _ontology_dict_setter(span: Span, value: Dict):
    span.doc.user_data[span] = value


def _ontology_dict_getter(span: Span):
    return span.doc.user_data.get(span, {})


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
        parser_name_to_entity_type: Dict[str, str],
    ):
        """

        :param span_key: the key for doc.spans to store the matches in
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

    def _load_curations(self, curated_list: PathLike) -> List[CuratedTerm]:
        with open(curated_list, mode="r") as jsonlf:
            curated_synonyms = [CuratedTerm(**json.loads(line)) for line in jsonlf]
        return curated_synonyms

    def _match_curations_to_ontology_terms(
        self, parsers: List[OntologyParser], curations: List[CuratedTerm]
    ) -> List[CuratedTerm]:

        case_sensitive: DefaultDict[str, DefaultDict[str, Set[SynonymTerm]]] = defaultdict(
            lambda: defaultdict(set)
        )
        case_insensitive: DefaultDict[str, DefaultDict[str, Set[SynonymTerm]]] = defaultdict(
            lambda: defaultdict(set)
        )
        matched_curations = []
        for parser in parsers:
            terms = parser.generate_synonyms()
            for term in terms:
                for term_string in term.terms:
                    case_sensitive[parser.entity_class][term_string].add(term)
                    case_insensitive[parser.entity_class][term_string.lower()].add(term)

        for curation in curations:
            if curation.action != "keep":
                logger.debug(f"dropping unwanted curation: {curation}")
                continue
            if curation.case_sensitive:
                query_dict = case_sensitive
                query_string = curation.term
            else:
                query_dict = case_insensitive
                query_string = curation.term.lower()

            matched_syn_terms = query_dict[curation.entity_class].get(query_string, set())
            if len(matched_syn_terms) == 0:
                logger.warning(f"failed to find ontology match for {curation}")
            else:

                non_redundant_syn_terms_by_parser = sort_then_group(
                    matched_syn_terms,
                    key_func=lambda x: (
                        x.parser_name,
                        x.associated_id_sets,
                    ),
                )
                for (
                    parser_name,
                    associated_id_sets,
                ), syn_terms_this_parser in non_redundant_syn_terms_by_parser:

                    # sort when choosing a term to use (amongst redundant terms) so that
                    # the chosen term is consistent between executions
                    syn_term_for_this_id_set = sorted(
                        syn_terms_this_parser, key=lambda x: x.term_norm
                    )[0]
                    curation.term_norm_mapping.setdefault(parser_name, set()).add(
                        syn_term_for_this_id_set.term_norm
                    )
                    if len(curation.term_norm_mapping[parser_name]) > 1:
                        logger.warning(
                            f"multiple SynonymTerm's detected for string {query_string}, "
                            f"This is probably means {query_string} is ambiguous in the "
                            f"parser {parser_name}"
                        )
            matched_curations.append(curation)

        return matched_curations

    def create_phrasematchers_from_curated_list(
        self, curated_list: PathLike, parsers: List[OntologyParser]
    ) -> Tuple[Optional[PhraseMatcher], Optional[PhraseMatcher]]:
        """
        in order to (non-redundantly) map a curated term to a :class:`.SynonymTerm`,
        we need to perform the following steps:

        1) look for the curated term string in the :attr:`.SynonymTerm.terms` field,
           checking for case sensitivity as we go

        2) for all matched :class:`.SynonymTerm`\\ s, check for redundancy by looking
           at the hash of :attr:`.SynonymTerm.associated_id_sets`

        3) for all non-redundant :class:`.SynonymTerm`\\ s, map the curated term string
           to the :attr:`.SynonymTerm.term_norm`

        :param curated_list:
        :param parsers:
        :return:
        """

        if self.strict_matcher is not None or self.lowercase_matcher is not None:
            logging.warning("Phrase matchers are being redefined - is this by intention?")
        self.set_labels(parser.entity_class for parser in parsers)
        curations = self._load_curations(curated_list)
        matched_curations = self._match_curations_to_ontology_terms(
            parsers=parsers, curations=curations
        )

        strict_matcher = PhraseMatcher(self.nlp.vocab, attr="ORTH")
        lowercase_matcher = PhraseMatcher(self.nlp.vocab, attr="NORM")

        patterns = self.nlp.tokenizer.pipe(
            # we need it lowercased for the case-insensitive matcher
            curation.term if curation.case_sensitive else curation.term.lower()
            for curation in matched_curations
        )

        for curation, pattern in zip(matched_curations, patterns):
            # a curation can have different term_norms for different parsers,
            # since the string normalizer's output depends on the entity class.
            # Also, a curation may exist in multiple SynonymTerm.terms
            for parser_name, term_norms in curation.term_norm_mapping.items():
                for term_norm in term_norms:
                    match_id = parser_name + self.match_id_sep + term_norm
                    if curation.case_sensitive:
                        strict_matcher.add(match_id, [pattern])
                    else:
                        lowercase_matcher.add(match_id, [pattern])

        # only set the phrasematcher if we have any rules for them
        # this lets us skip running a phrasematcher if it has no rules when we come
        # to calling OntologyMatcher
        self.strict_matcher = strict_matcher if len(strict_matcher) != 0 else None
        self.lowercase_matcher = lowercase_matcher if len(lowercase_matcher) != 0 else None
        return self.strict_matcher, self.lowercase_matcher

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
        self.set_labels(parser.entity_class for parser in parsers)
        lowercase_matcher = PhraseMatcher(self.nlp.vocab, attr="NORM")
        for parser in parsers:
            synonym_terms = parser.generate_synonyms()
            parser_name = parser.name
            logging.info(
                f"generating {sum(len(x.terms) for x in synonym_terms)} patterns for {parser_name}"
            )
            synonyms_and_terms = [
                (term, synonym_term)
                for synonym_term in synonym_terms
                for term in synonym_term.terms
            ]
            patterns = self.nlp.tokenizer.pipe(term for (term, _synonym_term) in synonyms_and_terms)

            for (term, synonym_term), pattern in zip(synonyms_and_terms, patterns):
                match_id = parser_name + self.match_id_sep + synonym_term.term_norm
                try:
                    # if we're adding to the lowercase matcher, we don't need to add
                    # to the exact case matcher as well, since we'll definitely get
                    # the hit, so would just be a waste of memory and compute.
                    lowercase_matcher.add(match_id, [pattern])

                except KeyError as e:
                    logging.warning(
                        f"failed to add '{term}'. StringStore is {len(self.nlp.vocab.strings)} ",
                        e,
                    )

        self.lowercase_matcher = lowercase_matcher

        if len(lowercase_matcher) == 0:
            raise RuntimeError(
                "No rules have been added to the PhraseMatcher. Has the OntologyMatcher been given parsers with synonyms?"
            )
        return lowercase_matcher, None

    def __call__(self, doc: Doc) -> Doc:
        if self.nr_strict_rules == 0 and self.nr_lowercase_rules == 0:
            raise ValueError("there are no matcher rules configured!")

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
        else:
            # this isn't possible since if both of them were None,
            # the if clause at the start of this function would raise an error
            raise AssertionError()

        spans = []
        for (start, end), matches_grp in sort_then_group(
            matches,
            key_func=lambda x: (
                x[1],
                x[2],
            ),
        ):
            data = defaultdict(set)
            for mat in matches_grp:
                parser_name, term_norm = self.nlp.vocab.strings.as_string(mat[0]).split(
                    self.match_id_sep, maxsplit=1
                )
                ent_class = self.parser_name_to_entity_type[parser_name]
                data[ent_class].add(
                    (
                        parser_name,
                        term_norm,
                    )
                )
            for ent_class in data:
                # we use a uuid here so that every span hash is unique
                new_span = Span(doc, start, end, label=uuid.uuid4().hex)
                new_span._.set("ontology_dict_", {ent_class: data[ent_class]})
                spans.append(new_span)
        final_spans = self.filter_by_contexts(doc, spans)
        span_group = SpanGroup(doc, name=self.span_key, spans=final_spans)
        doc.spans[self.span_key] = span_group
        return doc

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
                for ent_class in s._.ontology_dict_:
                    if ent_class != label and ent_class in self.labels:
                        self._set_token_attr(ent_class=ent_class, span=s, value=True)
            # Set the token attribute for each span of this label one-by-one, then match
            for span in spans:
                for ent_class in span._.ontology_dict_:
                    if ent_class == label:
                        self._set_token_attr(ent_class=ent_class, span=span, value=True)
                        context = doc
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

    def _set_token_attr(self, ent_class: str, span: Span, value: bool):
        for token in span:
            token._.set(ent_class, value)

    def span_in_TP_context(self, doc, ent_class: str):
        """When an entity type has a TP matcher defined, it should match for this
        span to be regarded as a true hit."""
        assert self.tp_matchers is not None
        tp_matcher = self.tp_matchers.get(ent_class, None)
        if tp_matcher:
            rule_spans = tp_matcher(doc, as_spans=True)
            if len(rule_spans) > 0:
                return True
            return False
        return True

    def span_in_FP_context(self, doc, ent_class: str):
        """When an entity type has a FP matcher defined, spans that match
        are regarded as FPs."""
        assert self.fp_matchers is not None
        fp_matcher = self.fp_matchers.get(ent_class, None)
        if fp_matcher:
            rule_spans = fp_matcher(doc, as_spans=True)
            if len(rule_spans) > 0:
                return True
            return False
        return False

    def span_in_TP_coocc(self, doc, span: Span, ent_class: str):
        """When an entity type has a TP co-occ dict defined, a hit defined in the dict
        is only regarded as a true hit when it matches at least one of its co-occ terms."""
        assert self.tp_coocc_dict is not None
        tp_dict = self.tp_coocc_dict.get(ent_class, None)
        if tp_dict and tp_dict.get(span.text):
            for w in tp_dict[span.text]:
                if w in doc.text:
                    return True
            return False
        return True

    def span_in_FP_coocc(self, doc, span: Span, ent_class: str):
        """When an entity type has a FP co-occ dic defined, a hit defined in the dict
        is regarded as a false positive when it matches at least one of its co-occ terms."""
        assert self.fp_coocc_dict is not None
        fp_dict = self.fp_coocc_dict.get(ent_class, None)
        if fp_dict and fp_dict.get(span.text):
            for w in fp_dict[span.text]:
                if w in doc.text:
                    return True
            return False
        return False

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
