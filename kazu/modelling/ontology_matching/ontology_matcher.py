from functools import partial
from typing import List, Dict, Union, Callable, Iterable, Optional, TypedDict
from pathlib import Path
import pandas as pd
import logging
import srsly
import pickle

import spacy
from spacy import Language
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import SpanGroup, Doc, Token
from spacy.util import SimpleFrozenList

from kazu.utils.utils import PathLike, SinglePathLikeOrIterable, as_path


STRICT_HITS = "spans_ORTH"
LOWER_HITS = "spans_LOWER"

GENE = "GGP"
CHEMICAL = "Chemical"
ANATOMY = "Anatomy"
DISEASE = "Disease"
CELL_LINE = "Cell_line"


@Language.factory(
    "ontology_matcher",
    default_config={
        "span_key": "RAW_HITS",
        "entry_filter": {"@misc": "arizona.entry_filter_blacklist.v1"},
        "variant_generator": {"@misc": "arizona.variant_generator.v1"},
    },
)
def create_ontology_mather(
    nlp, name, span_key: str, entry_filter: Callable, variant_generator: Callable
):
    return OntologyMatcher(
        nlp,
        name,
        span_key=span_key,
        entry_filter=entry_filter,
        variant_generator=variant_generator,
    )


# TODO: this can probably be written as an actual dataclass,
# but doing this for now until I get things working, and then I can try
# converting it.
class OntologyMatcherConfig(TypedDict):
    span_key: str
    labels: List[str]
    parquet_files: List[str]


class OntologyMatcher:
    def __init__(
        self,
        nlp: Language,
        name: str = "ontology_matcher",
        *,
        span_key: str = "RAW_HITS",
        entry_filter: Callable,
        variant_generator: Callable,
    ):
        """
        Create an OntologyMatcher.

        span_key (str): the key for doc.spans to store the matches in
        entry_filter (Callable): a function deciding whether a given ontology row/entry is valid
        variant_generator (Callable): a function generating variants for a given synonym
        """
        self.nlp = nlp
        self.name = name
        self.entry_filter = entry_filter
        self.variant_generator = variant_generator
        self.cfg: OntologyMatcherConfig = {"span_key": span_key, "labels": [], "parquet_files": []}
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
        return self.cfg["span_key"]

    @property
    def labels(self) -> List[str]:
        """RETURNS (List[str]): The labels currently processed by this component."""
        return self.cfg["labels"]

    def set_labels(self, labels: Iterable[str]):
        self.cfg["labels"] = list(labels)
        self.set_context_matchers()

    def set_context_matchers(self):
        self.tp_matchers, self.fp_matchers = self._create_token_matchers()
        self.tp_coocc_dict, self.fp_coocc_dict = self._create_coocc_dicts()

    @property
    def ontologies(self) -> List[str]:
        """RETURNS (List[str]): List of parquet files that were processed"""
        return self.cfg["parquet_files"]

    def set_ontologies(self, parquet_files: SinglePathLikeOrIterable):
        """Initialize the ontologies when creating this component.
        This method should not be run on an existing or deserialized pipeline.
        """
        if len(self.ontologies) > 0:
            logging.warning("Ontologies are being redefined - is this by intention?")
        if isinstance(parquet_files, (Path, str)):
            paths = self._define_paths(parquet_files)
        else:
            # TODO: refactor to make paths a Set[Path]
            paths = []
            for path in parquet_files:
                paths.extend(self._define_paths(path))

        dfs = {path.name: pd.read_parquet(path) for path in paths}
        self.strict_matcher = self._create_phrasematcher(dfs, lowercase=False)
        self.lowercase_matcher = self._create_phrasematcher(dfs, lowercase=True)
        self.cfg["parquet_files"] = [path.name for path in paths]

    def initialize(
        self,
        get_examples: Callable,
        *,
        nlp: Optional[Language] = None,
        parquet_files: SinglePathLikeOrIterable = None,
        labels: Optional[Iterable[str]] = None,
    ) -> None:
        """Initialize the labels and ontologies when creating this component.
        This method is not called when reading the pipeline from disk.

        nlp (Language): The current nlp object the component is part of.
        parquet_files (str or Path): location to parquet file or directory of parquet files
        labels (Optional[Iterable[str]]): The labels to add to the component
        """
        # Define the labels
        if not labels:
            ex_labels = set()
            for example in get_examples():
                for cat in example.y.cats:
                    ex_labels.add(cat)
            self.set_labels(ex_labels)
            if len(self.labels) > 0:
                logging.info(f"Inferred {len(self.labels)} labels from the data.")
            else:
                self.set_labels([GENE, CHEMICAL, ANATOMY, DISEASE, CELL_LINE])
                logging.info(f"Used the {len(self.labels)} default labels.")

        else:
            self.set_labels(labels)
            logging.info(f"Used the {len(self.labels)} given labels.")

        # Define the ontologies
        if parquet_files:
            self.set_ontologies(parquet_files)

    def __call__(self, doc: Doc) -> Doc:
        if self.nr_strict_rules == 0 or self.nr_lowercase_rules == 0:
            raise ValueError(
                "The matcher rules have not been set up properly. "
                "Did you initialize the labels and the ontologies?"
            )

        assert self.strict_matcher is not None
        assert self.lowercase_matcher is not None
        strict_matches = self._set_span_labels(self.strict_matcher(doc, as_spans=True))
        lower_matches = self._set_span_labels(self.lowercase_matcher(doc, as_spans=True))
        final_spans = self.resolve_conflicts(strict_matches, lower_matches)
        final_spans = self.filter_by_contexts(doc, final_spans)
        self.set_annotations(doc, final_spans)
        return doc

    def set_annotations(self, doc, spans):
        span_key = self.span_key
        span_group = SpanGroup(doc, name=span_key, spans=spans)
        doc.spans[span_key] = span_group

    def resolve_conflicts(self, strict_matches, lower_matches):
        # Alternative 1: add all strings together and filter on longest
        clean_spans, labeled_tokens = self._keep_longest(strict_matches + lower_matches)
        return clean_spans

        # Alternative 2: give preference over STRICT matching
        # Note that this will prefer "stroke" over "ischemic stroke" if the latter is matched lower-case
        # clean_spans, labeled_tokens = self._keep_longest(strict_matches)
        # clean_lower_spans, labeled_tokens = self._keep_longest(
        #     lower_matches, labeled_tokens
        # )
        # clean_spans.extend(clean_lower_spans)
        # return clean_spans

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

    def _keep_longest(self, orig_spans, labeled_tokens=None):
        """Keep only the longest non-overlapping spans"""
        final_spans = []
        if not labeled_tokens:
            labeled_tokens = set()

        for span in sorted(orig_spans, key=self._span_sort_key, reverse=True):
            overlap = False
            for token_i in range(span.start, span.end):
                if token_i in labeled_tokens:
                    overlap = True
            if not overlap:
                final_spans.append(span)
                for token_i in range(span.start, span.end):
                    labeled_tokens.add(token_i)
        return final_spans, labeled_tokens

    @staticmethod
    def _span_sort_key(span):
        return (span.end - span.start, -span.start)

    def _keep_longest_per_label(self, orig_spans, labeled_tokens=None):
        """Keep all longest spans per label - allowing one token to be part of different span types"""
        final_spans = []
        if not labeled_tokens:
            labeled_tokens = {}

        for span in sorted(orig_spans, key=self._span_sort_key, reverse=True):
            overlap = False
            current_label = span.label_
            for token_i in range(span.start, span.end):
                if current_label in labeled_tokens.get(token_i, set()):
                    overlap = True
            if not overlap:
                final_spans.append(span)
                for token_i in range(span.start, span.end):
                    new_set = labeled_tokens.get(token_i, set())
                    new_set.add(current_label)
                    labeled_tokens[token_i] = new_set
        return final_spans, labeled_tokens

    def _define_paths(self, in_loc: PathLike) -> List[Path]:
        in_path = as_path(in_loc)
        if not in_path.exists():
            raise ValueError(f"Location {in_loc} is not an existing file or directory.")
        if in_path.is_file():
            if in_path.suffix != ".parquet":
                raise ValueError(f"Provided file {in_loc} is not a Parquet file.")
            return [in_path]
        if in_path.is_dir():
            return [f for f in in_path.glob("**/*") if f.suffix == ".parquet"]
        return []

    def _set_span_labels(self, spans):
        for span in spans:
            iri = span.label
            span.label_ = self._label_from_IRI(span.label_)
            span.kb_id = iri
        return spans

    def _label_from_IRI(self, iri):
        """Return the correct span type, according to the given iri.
        Return an empty span if a certain ontology is not covered by the
        current set of labels."""
        if iri.startswith("ENS"):
            return GENE if GENE in self.labels else ""
        if iri.startswith("CHEMBL"):
            return CHEMICAL if CHEMICAL in self.labels else ""
        if iri.startswith("UBERON"):
            return ANATOMY if ANATOMY in self.labels else ""
        if iri.startswith("MONDO") or iri.startswith("HP"):
            return DISEASE if DISEASE in self.labels else ""
        if iri.startswith("CLO") or iri.startswith("CVCL"):
            return CELL_LINE if CELL_LINE in self.labels else ""
        raise ValueError(f"Can not deduce Ontology from IRI {iri}")

    def _create_phrasematcher(self, dfs, lowercase=False):
        attr = "ORTH"
        if lowercase:
            attr = "NORM"
        matcher = PhraseMatcher(self.nlp.vocab, attr=attr)
        for name, df in dfs.items():
            df = df[df.apply((lambda x: self.entry_filter(x, lowercase=lowercase)), axis=1)]
            terms = list([syn for syn in df["syn"]])
            iris = list(df["iri"])
            assert len(list(terms)) == len(list(iris))
            for iri, term in zip(iris, terms):
                if lowercase:
                    term = term.lower()
                variant_terms = list(self.variant_generator(term))
                patterns = list(self.nlp.tokenizer.pipe(variant_terms))
                matcher.add(iri, patterns)
        return matcher

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
        if CHEMICAL in self.labels:
            p = [{"_": {CHEMICAL: True}}, {"_": {ANATOMY: True}, "LOWER": "arm"}]
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

    def _create_gene_fp_dict(self):
        """Define cooccurrence links that determine likely FP gene hits"""
        gene_dict = {}
        gene_dict["IVF"] = ["ICSI", "cycle", "treatment"]
        return gene_dict

    def _create_disease_fp_dict(self):
        """Define cooccurrence links that determine likely FP disease hits"""
        disease_dict = {}
        disease_dict["MFS"] = ["endpoint"]
        return disease_dict

    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> None:
        """Serialize the pipe to disk.

        path (str / Path): Path to a directory.
        exclude (Iterable[str]): String names of serialization fields to exclude.
        """

        def pickle_matcher(p, strict):
            with p.open("wb") as outfile:
                if strict:
                    pickle.dump(self.strict_matcher, outfile)
                else:
                    pickle.dump(self.lowercase_matcher, outfile)

        serialize = {}
        serialize["cfg"] = lambda p: srsly.write_json(p, self.cfg)
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
        deserialize["cfg"] = lambda p: self.cfg.update(srsly.read_json(p))
        deserialize["strict_matcher"] = partial(unpickle_matcher, strict=True)
        deserialize["lowercase_matcher"] = partial(unpickle_matcher, strict=False)
        spacy.util.from_disk(path, deserialize, exclude)
        self.set_context_matchers()
        return self
