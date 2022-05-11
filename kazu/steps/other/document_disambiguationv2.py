import copy
import dataclasses
import functools
import itertools
import logging
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Set, Iterable, Callable, FrozenSet, DefaultDict, Dict

import numpy as np
import pydash
from kazu.data.data import (
    Document,
    Mapping,
    Entity,
    SynonymData,
    EquivalentIdAggregationStrategy,
)
from kazu.data.data import LinkRanks
from kazu.modelling.ontology_preprocessing.base import (
    DEFAULT_LABEL,
    MetadataDatabase,
    SynonymDatabase,
    StringNormalizer,
)
from kazu.steps import BaseStep
from kazu.utils.link_index import Hit, create_char_ngrams
from kazu.utils.spacy_pipeline import SpacyPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.tokens import Doc
from strsimpy import NGram, LongestCommonSubsequence
import pandas as pd

logger = logging.getLogger(__name__)

DISAMBIGUATED_BY = "disambiguated_by"
DISAMBIGUATED_BY_DEFINED_ELSEWHERE = "defined elsewhere in document"
DISAMBIGUATED_BY_REACTOME = "reactome pathway links in document"
DISAMBIGUATED_BY_CONTEXT = "document context"
KB_DISAMBIGUATION_FAILURE = "unable_to_disambiguate_within_ontology"
GLOBAL_DISAMBIGUATION_FAILURE = "unable_to_disambiguate_on_context"
SUGGEST_DROP = "suggest_drop"

UMAMBIGUOUS_SYNONYM_MERGE_STRATEGIES = {
    EquivalentIdAggregationStrategy.UNAMBIGUOUS,
    EquivalentIdAggregationStrategy.AMBIGUOUS_WITHIN_SINGLE_KB_MERGE,
    EquivalentIdAggregationStrategy.AMBIGUOUS_ACROSS_MULTIPLE_COMPOSITE_KBS_MERGE,
    EquivalentIdAggregationStrategy.AMBIGUOUS_WITHIN_SINGLE_KB_AND_ACROSS_MULTIPLE_COMPOSITE_KBS_MERGE,
}


class NgramEmbeddingStringSimilarityResolver:
    """
    attempt to score NON symbolic alternative names to an entity match string
    """

    def __init__(self, min_similarity_score: float = 0.85, min_lcs_length: float = 0.7):
        self.min_lcs_length = min_lcs_length
        self.min_similarity_score = min_similarity_score
        self.entity_match_string = None
        self.entity_match_string_spacy = None
        self.ent_match_norm = None
        self.number_resolver = None
        self.spacy_pipeline = SpacyPipeline(
            "/Users/test/PycharmProjects/azner/model_pack/scispacy/en_core_sci_md-0.4.0/en_core_sci_md/en_core_sci_md-0.4.0"
        )
        self.lcs = LongestCommonSubsequence()

    def clean_norm_string_for_embedding(self, norm_string: str):
        parts = [x for x in norm_string.lower().split(" ") if x != "s"]
        return " ".join(parts)

    @functools.lru_cache(maxsize=1)
    def prepare(self, entity_match_string: str):
        self.entity_match_string = entity_match_string
        self.ent_match_norm = StringNormalizer.normalize(entity_match_string)
        self.entity_match_string_spacy = self.spacy_pipeline.instance.nlp(
            self.clean_norm_string_for_embedding(self.ent_match_norm)
        )
        self.number_resolver = NumberResolver(self.ent_match_norm)
        self.min_lcs_score = float(len(self.ent_match_norm)) * self.min_lcs_length

    def score_alternative_name(
        self, entity_match_string: str, alternative_name_norm: str
    ) -> Dict[str, bool]:
        self.prepare(entity_match_string)

        if not self.number_resolver(alternative_name_norm):
            logger.debug(
                f"{alternative_name_norm} still ambiguous: number mismatch: {self.ent_match_norm}"
            )
            return {"number_correct": False}

        else:
            result = {"number_correct": True}

            lcs_score = self.lcs.length(alternative_name_norm, self.ent_match_norm)
            logger.info(f"lcs score: {lcs_score}: {self.ent_match_norm} -> {alternative_name_norm}")
            if lcs_score >= self.min_lcs_score:
                result["lcs_score"] = True
            else:
                result["lcs_score"] = False

            alternative_name_doc: Doc = self.spacy_pipeline.instance.nlp(
                self.clean_norm_string_for_embedding(alternative_name_norm)
            )

            if any(not token.has_vector for token in alternative_name_doc):
                logger.info(
                    f"cannot perform embedding similarity due to OOV for {alternative_name_norm}"
                )
                result["embedding_score"] = False
            else:
                simscore = self.entity_match_string_spacy.similarity(alternative_name_doc)
                logger.info(
                    f"embedding score: {simscore}: {self.entity_match_string_spacy.text} -> {alternative_name_doc.text}"
                )
                if simscore >= self.min_similarity_score:
                    result["embedding_score"] = True
                else:
                    result["embedding_score"] = False
        return result


class NumberCheckStringSimilarityResolver:
    @functools.lru_cache(maxsize=1)
    def prepare(self, entity_match_string: str):
        self.ent_match_norm = StringNormalizer.normalize(entity_match_string)
        self.number_resolver = NumberResolver(self.ent_match_norm)

    def score_alternative_name(self, entity_match_string: str, alternative_name_norm: str) -> Dict[str,bool]:
        self.prepare(entity_match_string)
        if not self.number_resolver(alternative_name_norm):
            logger.debug(
                f"{alternative_name_norm} still ambiguous: number mismatch: {self.ent_match_norm}"
            )
            return {"number_correct": False}
        else:
            return {"number_correct": True}


def is_probably_symbol_like(original_string: str) -> bool:
    # a more forgiving version of is_symbol_like, designed to improve symbol recall on natural text
    # 1 to prevent div zero
    upper_count = 1
    lower_count = 1
    int_count = 1

    for char in original_string:
        if char.isalpha():
            if char.isupper():
                upper_count += 1
            else:
                lower_count += 1
        elif char.isnumeric():
            int_count += 1

    upper_lower_ratio = float(upper_count) / float(lower_count)
    int_alpha_ration = float(int_count) / (float(upper_count + lower_count - 1))
    if upper_lower_ratio >= 1.0 or int_alpha_ration >= 1.0:
        return True
    else:
        return False


class SynonymDbQueryExtensions:
    def __init__(self):
        self.synonym_db = SynonymDatabase()

    def create_corpus_for_source(
        self, ambig_hits_this_source: List[Hit]
    ) -> Tuple[List[str], DefaultDict[str, Set[SynonymData]]]:
        corpus = []
        unambiguous_syn_to_syn_data = defaultdict(set)
        for hit in ambig_hits_this_source:
            for syn_data in hit.syn_data:
                for idx in syn_data.ids:
                    syns_this_id = self.synonym_db.get_syns_for_id(hit.parser_name, idx)
                    for syn_this_id in syns_this_id:
                        syn_data_this_syn = self.synonym_db.get(hit.parser_name, syn_this_id)
                        # ignore any ambiguous synonyms
                        if len(syn_data_this_syn) == 1:
                            corpus.append(syn_this_id)
                            unambiguous_syn_to_syn_data[syn_this_id].add(
                                next(iter(syn_data_this_syn))
                            )
        assert all(len(x) == 1 for x in unambiguous_syn_to_syn_data.values())
        # assert all(len(x)==1 for x in unambiguous_syn_to_original_string_norm.values())
        return corpus, unambiguous_syn_to_syn_data

    def collect_all_syns_from_ents(self, ents: List[Entity]) -> List[str]:
        result = []
        for ent in ents:
            for hit in ent.hits:
                if hit.confidence == LinkRanks.LOW_CONFIDENCE:
                    continue
                else:
                    for syn_data in hit.syn_data:
                        for idx in syn_data.ids:
                            result.extend(
                                pydash.flatten_deep(
                                    list(
                                        self.synonym_db.get_syns_for_id(
                                            name=hit.parser_name, idx=idx
                                        )
                                    )
                                )
                            )
        return result


class DocumentManipulator:
    def mappings_to_parser_name_and_idx_tuples(self, document: Document) -> Set[Tuple[str, str]]:
        ents = document.get_entities()
        result = set()
        for ent in ents:
            for mapping in ent.mappings:
                result.add((mapping.parser_name, mapping.idx))
        return result

    def get_document_representation(self, document: Document) -> List[str]:
        entities = document.get_entities()
        return [StringNormalizer.normalize(x.match) for x in entities]

    def __hash__(self):
        return id(self)


@functools.lru_cache(maxsize=1)
def build_query_matrix_cacheable(
    document: Document, vectorizer: TfidfVectorizer, manipulator: DocumentManipulator
):
    query = " . ".join(manipulator.get_document_representation(document))
    query_mat = vectorizer.transform([query]).todense()
    return query_mat


@dataclasses.dataclass
class DisambiguatedHit:
    original_hit: Hit
    mapping_type: FrozenSet[str]
    idx: str
    source: str
    confidence: LinkRanks
    parser_name: str


class KnowledgeBaseDisambiguationStrategy:
    def prepare(self, document: Document):
        pass

    def __call__(
        self, ent_match: str, document: Document, hits: List[Hit]
    ) -> Iterable[DisambiguatedHit]:
        raise NotImplementedError()


class PreferDefaultLabelKnowledgeBaseDisambiguationStrategy(KnowledgeBaseDisambiguationStrategy):
    """
    assumptions: default label is unambiguous within KB
    """

    def __init__(self):
        self.metadata_db = MetadataDatabase()
        self.default_label_norm_to_id = self._build_default_label_norm_lookup()

    def _build_default_label_norm_lookup(self):
        parser_names = self.metadata_db.get_loaded_parsers()
        default_label_norm_to_id = defaultdict(lambda: defaultdict(set))
        for parser_name in parser_names:
            all_metadata_this_parser = self.metadata_db.get_all(parser_name)
            for idx, metadata in all_metadata_this_parser.items():
                default_label_norm = StringNormalizer.normalize(metadata[DEFAULT_LABEL])
                default_label_norm_to_id[parser_name][default_label_norm].add(idx)
        return default_label_norm_to_id

    def __call__(
        self, ent_match: str, document: Document, hits: List[Hit]
    ) -> Iterable[DisambiguatedHit]:
        ent_match_norm = StringNormalizer.normalize(ent_match)
        found = False
        for hit in hits:
            default_label_ids = self.default_label_norm_to_id[hit.parser_name].get(
                ent_match_norm, set()
            )
            for syn_data in hit.syn_data:
                found_ids = default_label_ids.intersection(syn_data.ids)
                for idx in found_ids:
                    yield DisambiguatedHit(
                        original_hit=hit,
                        idx=idx,
                        source=syn_data.ids_to_source[idx],
                        confidence=hit.confidence,
                        parser_name=hit.parser_name,
                        mapping_type=syn_data.mapping_type,
                    )
                    found = True
                if found:
                    break
            if found:
                break


class RequireHighConfidenceKnowledgeBaseDisambiguationStrategy(KnowledgeBaseDisambiguationStrategy):
    def __call__(
        self, ent_match: str, document: Document, hits: List[Hit]
    ) -> Iterable[DisambiguatedHit]:
        for hit in hits:
            if hit.confidence == LinkRanks.HIGH_CONFIDENCE:
                for syn_data in hit.syn_data:
                    if syn_data.aggregated_by in UMAMBIGUOUS_SYNONYM_MERGE_STRATEGIES:
                        for idx in syn_data.ids:
                            yield DisambiguatedHit(
                                original_hit=hit,
                                idx=idx,
                                source=syn_data.ids_to_source[idx],
                                confidence=hit.confidence,
                                parser_name=hit.parser_name,
                                mapping_type=syn_data.mapping_type,
                            )


class RequireFullDefinitionKnowledgeBaseDisambiguationStrategy(KnowledgeBaseDisambiguationStrategy):
    def __init__(self):
        self.manipulator = DocumentManipulator()

    def __call__(
        self, ent_match: str, document: Document, hits: List[Hit]
    ) -> Iterable[DisambiguatedHit]:
        already_resolved_mappings_tup = self.manipulator.mappings_to_parser_name_and_idx_tuples(
            document
        )
        for hit in hits:
            for syn_data in hit.syn_data:
                for idx in syn_data.ids:
                    if (
                        hit.parser_name,
                        idx,
                    ) in already_resolved_mappings_tup:
                        for idx in syn_data.ids:
                            yield DisambiguatedHit(
                                original_hit=hit,
                                idx=idx,
                                source=syn_data.ids_to_source[idx],
                                confidence=LinkRanks.HIGH_CONFIDENCE,
                                parser_name=hit.parser_name,
                                mapping_type=syn_data.mapping_type,
                            )


class TfIdfKnowledgeBaseDisambiguationStrategy(KnowledgeBaseDisambiguationStrategy):
    def __init__(self, vectoriser: TfidfVectorizer):
        self.metadata_db = MetadataDatabase()
        self.vectoriser = vectoriser
        self.corpus_scorer = TfIdfCorpusScorer(vectoriser)
        self.queries = SynonymDbQueryExtensions()
        self.manipulator = DocumentManipulator()
        self.query_mat = None
        self.manipulator = DocumentManipulator()
        self.good_tuples = set()

    def prepare(self, document: Document):
        self.query_mat = build_query_matrix_cacheable(document, self.vectoriser, self.manipulator)

    @functools.lru_cache(maxsize=1)
    def find_good_tuples(self, document: Document):
        self.good_tuples = self.manipulator.mappings_to_parser_name_and_idx_tuples(document)

    def get_string_resolver_strategy(self):
        raise NotImplementedError()

    def previously_found(self, parser_name, syn_data: SynonymData):
        return any(
            (
                parser_name,
                idx,
            )
            in self.good_tuples
            for idx in syn_data.ids
        )

    def __call__(
        self, ent_match: str, document: Document, hits: List[Hit]
    ) -> Iterable[DisambiguatedHit]:
        parser_name = hits[0].parser_name
        # todo: move to caching step
        self.find_good_tuples(document)
        string_resolver = self.get_string_resolver_strategy()
        # group hits by confidence (process high first)
        hits_by_confidence = itertools.groupby(
            sorted(hits, key=lambda x: x.confidence), key=lambda x: x.confidence
        )
        hit_found = False
        for confidence, hits_iter in hits_by_confidence:
            if hit_found:
                break

            hits_this_confidence = list(hits_iter)

            (
                unambiguous_synonym_corpus,
                unambiguous_syn_to_syn_data,
            ) = self.queries.create_corpus_for_source(hits_this_confidence)
            unambiguous_syns_and_scores = list(
                self.corpus_scorer(unambiguous_synonym_corpus, self.query_mat)
            )
            unambiguous_syns_and_scores = sorted(
                unambiguous_syns_and_scores, key=lambda x: x[1], reverse=True
            )
            for unambiguous_syn, score in unambiguous_syns_and_scores:
                # TODO: set his correctly
                if score <= 7.0:
                    break

                target_syn_data_set = unambiguous_syn_to_syn_data[unambiguous_syn]
                if len(target_syn_data_set) != 1:
                    logger.warning("target syndata is too long!")
                target_syn_data = next(iter(target_syn_data_set))
                previously_found = self.previously_found(parser_name, target_syn_data)
                if not previously_found:
                    string_score_results = string_resolver.score_alternative_name(
                        alternative_name_norm=unambiguous_syn, entity_match_string=ent_match
                    )
                else:
                    string_score_results = {}

                logger.info(
                    f"original: <{ent_match}>, matched: <{unambiguous_syn}>, previously found: <{previously_found}>, context: {score}, string score {string_score_results}"
                )
                if previously_found or all(x for x in string_score_results.values()):
                    hit_found = True
                    for idx in target_syn_data.ids:
                        yield DisambiguatedHit(
                            original_hit=None,
                            idx=idx,
                            source=target_syn_data.ids_to_source[idx],
                            confidence=LinkRanks.MEDIUM_CONFIDENCE, #set to medium, since we got to this strategy without finding a hit...
                            parser_name=parser_name,
                            mapping_type=target_syn_data.mapping_type,
                        )
                    # we break the loop after the first successful hit is found in this strategy, so as to not
                    # produce less good mappings than the best found
                    if hit_found:
                        logger.info(f"hit found at {confidence.name}. Stopping search")
                        break


class IntegerValidationTfIdfKnowledgeBaseDisambiguationStrategy(
    TfIdfKnowledgeBaseDisambiguationStrategy
):
    def get_string_resolver_strategy(self):
        return NumberCheckStringSimilarityResolver()


class NgramEmbeddingValidationTfIdfKnowledgeBaseDisambiguationStrategy(
    TfIdfKnowledgeBaseDisambiguationStrategy
):
    def get_string_resolver_strategy(self):
        return NgramEmbeddingStringSimilarityResolver()


class GlobalDisambiguationStrategy:
    def prepare(self, document: Document):
        pass

    def __call__(
        self, ent_match: str, entities: List[Entity], document: Document
    ) -> Tuple[List[Entity], List[Entity]]:
        raise NotImplementedError()


class RequireFullDefinitionGlobalDisambiguationStrategy(GlobalDisambiguationStrategy):
    def __init__(self):
        self.manipulator = DocumentManipulator()

    def __call__(
        self, ent_match: str, entities: List[Entity], document: Document
    ) -> Tuple[List[Entity], List[Entity]]:
        already_resolved_mappings_tup = self.manipulator.mappings_to_parser_name_and_idx_tuples(
            document
        )
        ents_defined_elsewhere = []
        ents_not_defined_elsewhere = []
        for ent in entities:
            hit_found = False
            for hit in ent.hits:
                for syn_data in hit.syn_data:
                    for idx in syn_data.ids:
                        if (
                            hit.parser_name,
                            idx,
                        ) in already_resolved_mappings_tup:
                            hit_found = True
                            break
                    if hit_found:
                        break
                if hit_found:
                    break
            if hit_found:
                ents_defined_elsewhere.append(ent)
            else:
                ents_not_defined_elsewhere.append(ent)

        return ents_defined_elsewhere, ents_not_defined_elsewhere


class KeepHighConfidenceHitsGlobalDisambiguationStrategy(GlobalDisambiguationStrategy):
    def __init__(self, min_string_length_to_test_for_high_confidence: int = 3):
        self.min_string_length_to_test_for_high_confidence = (
            min_string_length_to_test_for_high_confidence
        )

    def __call__(
        self, ent_match: str, entities: List[Entity], document: Document
    ) -> Tuple[List[Entity], List[Entity]]:
        ents_with_high_conf = []
        ents_without_high_conf = []
        if len(ent_match) >= self.min_string_length_to_test_for_high_confidence:
            for ent in entities:
                for hit in ent.hits:
                    if hit.confidence == LinkRanks.HIGH_CONFIDENCE:
                        ents_with_high_conf.append(ent)
                        break
                else:
                    ents_without_high_conf.append(ent)
        else:
            ents_without_high_conf = entities
        return ents_with_high_conf, ents_without_high_conf


class TfIdfGlobalDisambiguationStrategy(GlobalDisambiguationStrategy):
    def __init__(self, vectoriser: TfidfVectorizer, kbs_are_compatible: Callable[[Set[str]], bool]):
        self.kbs_are_compatible = kbs_are_compatible
        self.queries = SynonymDbQueryExtensions()
        self.vectoriser = vectoriser
        self.corpus_scorer = TfIdfCorpusScorer(vectoriser)
        self.threshold = 7.0
        # if any entity string matches  are longer than this and have a high confidence hit, assume they're probably right
        self.manipulator = DocumentManipulator()
        self.query_mat = None

    def prepare(self, document: Document):
        self.query_mat = build_query_matrix_cacheable(document, self.vectoriser, self.manipulator)

    def __call__(
        self, ent_match: str, entities: List[Entity], document: Document
    ) -> Tuple[List[Entity], List[Entity]]:
        # test context for good hits...
        disambiguated_ents = []
        still_ambiguous_ents = []
        # group ents by hit source
        hit_sources_to_ents = defaultdict(set)
        for ent in entities:
            for hit in ent.hits:
                hit_sources_to_ents[hit.parser_name].add(ent)

        corpus = list(set(self.queries.collect_all_syns_from_ents(entities)))
        for synonym, score in self.corpus_scorer(corpus, self.query_mat):
            if score < self.threshold:
                # no ents could be resolved this match
                still_ambiguous_ents.extend(entities)
                break

            kbs_this_hit = self.queries.synonym_db.get_kbs_for_syn_global(synonym)
            if not self.kbs_are_compatible(kbs_this_hit):
                logger.debug(f"class still ambiguous: {kbs_this_hit}")
            else:
                # for the first unambiguous kb hit,return the ent that has a hit with this kb.
                for kb_hit in kbs_this_hit:
                    ents_this_kb = hit_sources_to_ents.get(kb_hit)
                    if ents_this_kb:
                        disambiguated_ents.extend(list(ents_this_kb))
                        break
                if len(disambiguated_ents) > 0:
                    break
        return disambiguated_ents, still_ambiguous_ents


class GlobalDisambiguationStrategyList:
    def __init__(self, strategies: List[GlobalDisambiguationStrategy]):
        self.strategies = strategies
        self.metadata_db = MetadataDatabase()

    def __call__(
        self, entity_string: str, entities: List[Entity], document: Document
    ) -> Tuple[str, List[Entity], List[Entity]]:

        for strategy in self.strategies:
            ents_to_keep, still_ambiguous_ents = strategy(
                ent_match=entity_string, entities=entities, document=document
            )
            if len(ents_to_keep) > 0:
                return strategy.__class__.__name__, ents_to_keep, still_ambiguous_ents
        return "no_successful_strategy", [], entities


def create_char_3grams(string):
    return create_char_ngrams(string, n=3)


def create_word_ngrams(string: str, n=2):
    words = string.split(" ")
    ngrams = zip(*[words[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


def create_word_unigram_bigram_and_char_3grams(string):
    result = []
    unigrams = create_word_ngrams(string, 1)
    result.extend(unigrams)
    bigrams = create_word_ngrams(string, 2)
    result.extend(bigrams)
    char_trigrams = create_char_3grams(string)
    result.extend(char_trigrams)
    return result


class Disambiguator:
    """
    global:
    prefer any ent with an exact hit (disregard others if exact hit found)

    per class symbol strategy:
    disease: require full definition
    drug: require full definition



    needs to
    a) resolve all unambiguous non symbol like (e.g. noun phrases) via exact match no magic number
    b) resolve all ambiguous non symbol like (e.g. noun phrases) magic number
    b) identify all symbol like ents -> always disambiguate
    c) resolve all amb symbol like via unamb like no magic number
    d) resolve all remaining symbol like via TFIDF magic numbers



    """

    def __init__(self, path: str):
        self.syn_db = SynonymDatabase()
        self.metadata_db = MetadataDatabase()
        self.vectoriser: TfidfVectorizer = self.build_or_load_vectoriser(path)
        self.allowed_overlaps = {
            frozenset({"MONDO", "MEDDRA", "OPENTARGETS_DISEASE"}),
            frozenset({"CHEMBL", "OPENTARGETS_MOLECULE", "OPENTARGETS_TARGET"}),
        }
        self.always_disambiguate = {"ExplosionNERStep"}

        # strategies to implement:
        # found exact unambiguous match
        tfidf_symbolic_kb_strategy = IntegerValidationTfIdfKnowledgeBaseDisambiguationStrategy(
            self.vectoriser
        )
        tfidf_non_symbolic_kb_strategy = (
            NgramEmbeddingValidationTfIdfKnowledgeBaseDisambiguationStrategy(self.vectoriser)
        )
        required_full_definition_strategy = (
            RequireFullDefinitionKnowledgeBaseDisambiguationStrategy()
        )
        required_high_confidence = RequireHighConfidenceKnowledgeBaseDisambiguationStrategy()
        prefer_default_label = PreferDefaultLabelKnowledgeBaseDisambiguationStrategy()

        keep_high_conf_global_strategy = KeepHighConfidenceHitsGlobalDisambiguationStrategy()
        tfidf_global_strategy = TfIdfGlobalDisambiguationStrategy(
            self.vectoriser, self.kbs_are_compatible
        )
        # embedding_kb_strategy = PreferEmbeddingKnowledgeBaseDisambiguationStrategy(
        #     set("SapBertForEntityLinkingStep")
        # )
        # embedding_kb_strategy = EmbeddingKnowledgeBaseDisambiguationStrategy()

        self.default_strategy = [required_high_confidence]
        self.symbolic_disambiguation_strategy_lookup = {
            "gene": [
                required_full_definition_strategy,
                required_high_confidence,
                prefer_default_label,
                tfidf_symbolic_kb_strategy,
            ],
            "disease": [
                required_full_definition_strategy,
                required_high_confidence,
                prefer_default_label,
            ],
            "drug": [
                required_full_definition_strategy,
                required_high_confidence,
                prefer_default_label,
                tfidf_symbolic_kb_strategy,
            ],
            "anatomy": [
                required_high_confidence,
                prefer_default_label,
                tfidf_symbolic_kb_strategy,
            ],
        }
        self.non_symbolic_disambiguation_strategy_lookup = {
            "gene": [
                required_high_confidence,
                prefer_default_label,
                tfidf_non_symbolic_kb_strategy,
            ],
            "disease": [
                required_high_confidence,
                prefer_default_label,
                tfidf_non_symbolic_kb_strategy,
            ],
            "drug": [
                required_high_confidence,
                prefer_default_label,
                tfidf_non_symbolic_kb_strategy,
            ],
            "anatomy": [
                required_high_confidence,
                prefer_default_label,
                tfidf_non_symbolic_kb_strategy,
            ],
        }

        self.global_non_symbolic_disambiguation_strategy = GlobalDisambiguationStrategyList(
            [keep_high_conf_global_strategy, tfidf_global_strategy]
        )

        required_full_definition_global = RequireFullDefinitionGlobalDisambiguationStrategy()
        self.global_symbolic_disambiguation_strategy = GlobalDisambiguationStrategyList(
            [required_full_definition_global, tfidf_global_strategy]
        )

        self.all_strategies = [
            tfidf_symbolic_kb_strategy,
            tfidf_non_symbolic_kb_strategy,
            required_full_definition_strategy,
            keep_high_conf_global_strategy,
            tfidf_global_strategy,
            required_full_definition_global,
            required_high_confidence,
            # embedding_kb_strategy,
            prefer_default_label,
        ]

    def build_or_load_vectoriser(self, path_str: str):
        path = Path(path_str)
        if path.exists():
            return pickle.load(open(path, "rb"))
        else:
            vec = TfidfVectorizer(
                lowercase=False, analyzer=create_word_unigram_bigram_and_char_3grams
            )
            x = []
            for kb in self.syn_db.get_loaded_kbs():
                x.extend(list(self.syn_db.get_all(kb).keys()))
            vec.fit(x)
            pickle.dump(vec, open(path, "wb"))
            return vec

    def kbs_are_compatible(self, kbs: Set[str]):
        if len(kbs) == 1:
            return True
        for allowed_overlap in self.allowed_overlaps:
            if kbs.issubset(allowed_overlap):
                return True
        return False

    def sort_entities_by_symbolism(self, entities: List[Entity]):
        symbolic, non_symbolic = [], []
        grouped_by_match = itertools.groupby(
            sorted(
                entities,
                key=lambda x: (x.match),
            ),
            key=lambda x: (x.match),
        )

        for match_str, ent_iter in grouped_by_match:
            if is_probably_symbol_like(match_str):
                symbolic.extend(list(ent_iter))
            else:
                non_symbolic.extend(list(ent_iter))
        return symbolic, non_symbolic

    def prepare_strategies(self, doc: Document):
        for strategy in self.all_strategies:
            strategy.prepare(doc)

    def run(self, doc: Document):
        # TODO: cache any strategy data that only needs to be run once
        self.prepare_strategies(doc)
        entities = doc.get_entities()
        symbolic_entities, non_symbolic_entities = self.sort_entities_by_symbolism(entities)
        globally_disambiguated_non_symbolic_entities = self.execute_global_disambiguation_strategy(
            non_symbolic_entities, doc, False
        )
        self.execute_kb_disambiguation_strategy(
            globally_disambiguated_non_symbolic_entities, doc, False
        )

        globally_disambiguated_symbolic_entities = self.execute_global_disambiguation_strategy(
            symbolic_entities, doc, True
        )
        self.execute_kb_disambiguation_strategy(globally_disambiguated_symbolic_entities, doc, True)

    def execute_global_disambiguation_strategy(
        self, entities: List[Entity], document: Document, symbolic: bool
    ):
        result = []
        ents_by_match = itertools.groupby(
            sorted(entities, key=lambda x: x.match), key=lambda x: x.match
        )
        for match_str, ent_iter in ents_by_match:
            match_ents = list(ent_iter)
            if symbolic:
                (
                    strategy_used,
                    resolved_ents,
                    still_ambiguous_ents,
                ) = self.global_symbolic_disambiguation_strategy(match_str, match_ents, document)
            else:
                (
                    strategy_used,
                    resolved_ents,
                    still_ambiguous_ents,
                ) = self.global_non_symbolic_disambiguation_strategy(
                    match_str, match_ents, document
                )
            logger.info(
                f"global disambiguation of {match_str} with {strategy_used}: {len(resolved_ents)} references resolved"
            )
            result.extend(resolved_ents)
        return result

    def execute_kb_disambiguation_strategy(
        self, ents_needing_disambig: List[Entity], document: Document, symbolic: bool
    ):
        """


        get a list of add potential ids this ent, from the hits
        build corpus and search for most appropriate synoym as per normal
        perform hit post processing to score best matches -> if all fails threshold, choose sapbert result (if not symbolic)
        TODO:
        update document map of id -> result. If Id



        :param entitites:
        :param query_mat:
        :return:
        """
        if symbolic:
            strategy_map = self.symbolic_disambiguation_strategy_lookup
        else:
            strategy_map = self.non_symbolic_disambiguation_strategy_lookup

        def group_ents(
            ents: List[Entity],
        ) -> Iterable[Tuple[str, str, str, List[Entity], List[Hit]]]:
            ents_grouped_by_class_and_match = itertools.groupby(
                sorted(
                    ents,
                    key=lambda x: (
                        x.entity_class,
                        x.match,
                    ),
                ),
                key=lambda x: (x.entity_class, x.match),
            )
            for (entity_class, entity_match), ents_iter in ents_grouped_by_class_and_match:
                entities_this_class_and_match = list(ents_iter)
                # note,assume all ents with same match have same hits!
                hits = entities_this_class_and_match[0].hits
                hits_by_parser = itertools.groupby(
                    sorted(hits, key=lambda x: x.parser_name), key=lambda x: x.parser_name
                )
                for parser_name, hits_iter in hits_by_parser:
                    yield entity_class, entity_match, parser_name, entities_this_class_and_match, list(
                        hits_iter
                    )

        # entity_class, entity_match, parser_name,hits
        tuples: List[Tuple[str, str, str, List[Entity], List[Hit]]] = list(
            group_ents(ents_needing_disambig)
        )
        strategy_max_index = max(len(strategies) for strategies in strategy_map.values())
        match_and_parser_resolved = set()

        for i in range(0, strategy_max_index):
            for (
                entity_class,
                entity_match,
                parser_name,
                entities_this_group,
                hits_by_parser,
            ) in tuples:
                if (
                    entity_match,
                    parser_name,
                ) not in match_and_parser_resolved:

                    strategy_list = strategy_map.get(entity_class, self.default_strategy)
                    if i > len(strategy_list) - 1:
                        logger.debug("no more strategies this class")
                        continue
                    else:
                        strategy = strategy_list[i]
                        logger.info(
                            f"running strategy {strategy.__class__.__name__} on class :<{entity_class}>, match: <{entity_match}> for parser: {parser_name} remaining: {len(entities_this_group)}"
                        )
                        new_mappings = []
                        for disambiguate_hit in strategy(
                            ent_match=entity_match,
                            hits=hits_by_parser,
                            document=document,
                        ):
                            successful_strategy = strategy.__class__.__name__
                            additional_metadata = {DISAMBIGUATED_BY: successful_strategy}
                            # todo - calculate disambiguated conf better!
                            mapping = self.metadata_db.create_mapping(
                                parser_name=disambiguate_hit.parser_name,
                                source=disambiguate_hit.source,
                                idx=disambiguate_hit.idx,
                                mapping_type=disambiguate_hit.mapping_type,
                                confidence=disambiguate_hit.confidence,
                                additional_metadata=additional_metadata,
                            )
                            logger.info(
                                f"mapping created: original string: {entity_match}, mapping: {mapping}"
                            )
                            new_mappings.append(mapping)
                        if len(new_mappings) > 0:
                            for ent in entities_this_group:
                                ent.mappings.extend(copy.deepcopy(new_mappings))
                            match_and_parser_resolved.add(
                                (
                                    entity_match,
                                    parser_name,
                                )
                            )


def ent_match_group_key(ent: Entity):
    return ent.match, ent.entity_class


def mapping_kb_group_key(mapping: Mapping):
    return mapping.source


class DocumentLevelDisambiguationStep(BaseStep):

    """
    algorithm:

    there are two scenarios:

    a) entities with better than low confidence mappings, but are ambiguous
    b) entities with only low confidence mappings that are unambiguous

    """

    def __init__(
        self,
        depends_on: Optional[List[str]],
        tfidf_disambiguator: Disambiguator,
    ):
        """

        :param depends_on:
        :param tfidf_disambiguator: Disambiguator instance
        """

        super().__init__(depends_on)
        self.tfidf_disambiguator = tfidf_disambiguator

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs: List[Document] = []
        for doc in docs:
            self.tfidf_disambiguator.run(doc)
        return docs, failed_docs


class NumberResolver:
    number_finder = re.compile("[0-9]+")

    def __init__(self, query_string_norm):
        self.ent_match_number_count = Counter(re.findall(self.number_finder, query_string_norm))

    def __call__(self, synonym_string_norm: str):
        synonym_string_norm_match_number_count = Counter(
            re.findall(self.number_finder, synonym_string_norm)
        )
        return synonym_string_norm_match_number_count == self.ent_match_number_count


class SubStringResolver:
    def __init__(self, query_string_norm):
        self.query_string_norm = query_string_norm
        # require min 70% subsequence overlap
        self.min_distance = float(len(query_string_norm)) * 0.7
        self.lcs = LongestCommonSubsequence()

    def __call__(self, synonym_string_norm: str):
        length = self.lcs.distance(self.query_string_norm, synonym_string_norm)
        return length >= self.min_distance


class TfIdfCorpusScorer:
    def __init__(self, vectoriser: TfidfVectorizer):
        self.vectoriser = vectoriser

    def __call__(self, corpus: List[str], query_mat: np.ndarray) -> Iterable[Tuple[str, float]]:
        if len(corpus) == 0:
            return None
        else:
            neighbours, scores = self.find_neighbours_and_scores(corpus=corpus, query=query_mat)
            for neighbour, score in zip(neighbours, scores):
                synonym = corpus[neighbour]
                yield synonym, score

    def find_neighbours_and_scores(self, corpus: List[str], query: np.ndarray):
        mat = self.vectoriser.transform(corpus)
        score_matrix = np.squeeze(-np.asarray(mat.dot(query.T)))
        neighbours = score_matrix.argsort()
        if neighbours.size == 1:
            neighbours = np.array([0])
            distances = np.array([score_matrix.item()])
        else:
            distances = score_matrix[neighbours]
            distances = 100 * -distances
        return neighbours, distances


class EmbeddingCorpusScorer:
    def __init__(self, spacy_pipeline: SpacyPipeline):
        self.spacy_pipeline = spacy_pipeline

    def __call__(self, corpus: List[str], query: Doc) -> Iterable[Tuple[str, float]]:
        if len(corpus) == 0:
            return None
        else:
            for doc, score in self.find_neighbours_and_scores(corpus=corpus, query=query):
                yield doc.text, score

    def find_neighbours_and_scores(self, corpus: List[str], query: Doc):
        corpus_docs = self.spacy_pipeline.instance.nlp.pipe(corpus)
        corpus_docs_and_score = [
            (
                doc,
                query.similarity(doc),
            )
            for doc in corpus_docs
        ]
        return sorted(corpus_docs_and_score, key=lambda x: x[1], reverse=True)
