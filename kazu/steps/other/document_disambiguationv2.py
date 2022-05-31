import copy
import dataclasses
import functools
import itertools
import logging
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Set, Iterable, Callable, FrozenSet, DefaultDict, Dict

import numpy as np
import pandas as pd
import pydash
from kazu.data.data import (
    Document,
    Mapping,
    Entity,
    SynonymData,
    EquivalentIdAggregationStrategy,
    IS_SUBSPAN,
)
from kazu.data.data import LinkRanks, SearchRanks
from kazu.modelling.ontology_preprocessing.base import (
    DEFAULT_LABEL,
    MetadataDatabase,
    SynonymDatabase,
    StringNormalizer,
)
from kazu.modelling.ontology_preprocessing.synonym_generation import GreekSymbolSubstitution
from kazu.steps import BaseStep
from kazu.utils.link_index import Hit, create_char_ngrams, NumberResolver
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from strsimpy import LongestCommonSubsequence, NGram

logger = logging.getLogger(__name__)

DISAMBIGUATED_BY = "disambiguated_by"
DISAMBIGUATED_BY_DEFINED_ELSEWHERE = "defined elsewhere in document"
DISAMBIGUATED_BY_REACTOME = "reactome pathway links in document"
DISAMBIGUATED_BY_CONTEXT = "document context"
KB_DISAMBIGUATION_FAILURE = "unable_to_disambiguate_within_ontology"
GLOBAL_DISAMBIGUATION_FAILURE = "unable_to_disambiguate_on_context"
SUGGEST_DROP = "suggest_drop"
MATCHED_NUMBER_SCORE = "matched_number_score"
NGRAM_SCORE = "ngram_score"
FUZZ_SCORE = "fuzz_score"
UMAMBIGUOUS_SYNONYM_MERGE_STRATEGIES = {
    EquivalentIdAggregationStrategy.UNAMBIGUOUS,
    EquivalentIdAggregationStrategy.AMBIGUOUS_WITHIN_SINGLE_KB_MERGE,
    EquivalentIdAggregationStrategy.AMBIGUOUS_ACROSS_MULTIPLE_COMPOSITE_KBS_MERGE,
    EquivalentIdAggregationStrategy.AMBIGUOUS_WITHIN_SINGLE_KB_AND_ACROSS_MULTIPLE_COMPOSITE_KBS_MERGE,
}


class HitPostProcessor:
    def __init__(self):
        self.ngram = NGram(2)
        self.numeric_class_phrase_disambiguation = ["TYPE"]
        self.numeric_class_phrase_disambiguation_re = [
            re.compile(x + " [0-9]+") for x in self.numeric_class_phrase_disambiguation
        ]
        self.modifier_phrase_disambiguation = ["LIKE"]

    def phrase_disambiguation_filter(self, hits, text):
        new_hits = []
        for numeric_phrase_re in self.numeric_class_phrase_disambiguation_re:
            match = re.search(numeric_phrase_re, text)
            if match:
                found_string = match.group()
                for hit in hits:
                    if found_string in hit.string_norm:
                        new_hits.append(hit)
        if not new_hits:
            for modifier_phrase in self.modifier_phrase_disambiguation:
                in_text = modifier_phrase in text
                if in_text:
                    for hit in filter(lambda x: modifier_phrase in x.string_norm, hits):
                        new_hits.append(hit)
                else:
                    for hit in filter(lambda x: modifier_phrase not in x.string_norm, hits):
                        new_hits.append(hit)
        if new_hits:
            return new_hits
        else:
            return hits

    def ngram_scorer(self, hits: List[Hit], text):
        # low conf
        for hit in hits:
            hit.metrics[NGRAM_SCORE] = 2 / (self.ngram.distance(text, hit.string_norm) + 1.0)
        return hits

    def run_fuzz_algo(self, hits: List[Hit], text):
        # low conf
        choices = [x.string_norm for x in hits]
        if len(text) > 10 and len(text.split(" ")) > 4:
            scores = process.extract(text, choices, scorer=fuzz.token_sort_ratio)
        else:
            scores = process.extract(text, choices, scorer=fuzz.WRatio)
        for score in scores:
            hit = hits[score[2]]
            hit.metrics[FUZZ_SCORE] = score[1]
        return hits

    def run_number_algo(self, hits, text):
        number_resolver = NumberResolver(text)
        for hit in hits:
            numbers_matched = number_resolver(hit.string_norm)
            hit.metrics[MATCHED_NUMBER_SCORE] = numbers_matched
        return hits

    def __call__(self, hits: List[Hit], string_norm: str) -> List[Hit]:

        hits = self.phrase_disambiguation_filter(hits, string_norm)
        hits = self.run_number_algo(hits, string_norm)
        hits = self.run_fuzz_algo(hits, string_norm)
        hits = self.ngram_scorer(hits, string_norm)
        return hits


class NumberCheckStringSimilarityResolver:
    @functools.lru_cache(maxsize=1)
    def prepare(self, entity_match_string: str):
        self.ent_match_norm = StringNormalizer.normalize(entity_match_string)
        self.number_resolver = NumberResolver(self.ent_match_norm)

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
                for syn_data in hit.syn_data:
                    for idx in syn_data.ids:
                        result.extend(
                            pydash.flatten_deep(
                                list(self.synonym_db.get_syns_for_id(name=hit.parser_name, idx=idx))
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
        match_norm = StringNormalizer.normalize(ent_match)
        found = False
        for hit in hits:
            default_label_ids = self.default_label_norm_to_id[hit.parser_name].get(
                match_norm, set()
            )
            for syn_data in hit.syn_data:
                found_ids = default_label_ids.intersection(syn_data.ids)
                for idx in found_ids:
                    yield DisambiguatedHit(
                        original_hit=hit,
                        idx=idx,
                        source=syn_data.ids_to_source[idx],
                        confidence=LinkRanks.MEDIUM_HIGH_CONFIDENCE,
                        parser_name=hit.parser_name,
                        mapping_type=syn_data.mapping_type,
                    )
                    found = True
                if found:
                    break
            if found:
                break


class HitEnsembleKnowledgeBaseDisambiguationStrategy(KnowledgeBaseDisambiguationStrategy):
    def __init__(
        self,
        preference: str = "embedding",
        search_threshold=80.0,
        embedding_threshold=99.9945,
        embedding_threshold_max=99.9954,
        search_field="search_score",
        embedding_field="sapbert_score",
        max_rank_to_consider: float = 2.0,
    ):

        self.embedding_threshold_max = embedding_threshold_max
        self.embedding_field = embedding_field
        self.search_field = search_field
        self.embedding_threshold = embedding_threshold
        self.search_threshold = search_threshold
        self.preference = preference
        self.max_rank_to_consider = max_rank_to_consider
        self.hit_post_processor = HitPostProcessor()

    def hit_threshold_condition(
        self, search_score: float, embedding_score: float, rank_position: float
    ) -> Tuple[bool, LinkRanks]:

        if (
            rank_position == 1.0
            and search_score >= self.search_threshold
            and embedding_score >= self.embedding_threshold
        ):
            return True, LinkRanks.HIGH_CONFIDENCE
        elif (
            rank_position <= self.max_rank_to_consider
            and search_score >= self.search_threshold
            and embedding_score >= self.embedding_threshold
        ):
            return True, LinkRanks.MEDIUM_HIGH_CONFIDENCE
        elif (
            rank_position <= self.max_rank_to_consider
            and embedding_score >= self.embedding_threshold_max
        ):
            return True, LinkRanks.MEDIUM_CONFIDENCE
        else:
            return False, LinkRanks.LOW_CONFIDENCE

    def __call__(
        self, ent_match: str, document: Document, hits: List[Hit]
    ) -> Iterable[DisambiguatedHit]:
        records = []
        string_norm = StringNormalizer.normalize(ent_match)
        hits = self.hit_post_processor(hits, string_norm)
        # only consider hits with matched numbers
        hits = [x for x in hits if x.metrics[MATCHED_NUMBER_SCORE]]
        hits_by_syn_data = {
            k: set(v)
            for k, v in itertools.groupby(
                sorted(hits, key=lambda x: tuple(x.syn_data)), key=lambda x: tuple(x.syn_data)
            )
        }

        for syn_data, hit_set in hits_by_syn_data.items():
            record = {"syn_data": syn_data}
            global_metrics = {}
            # choose best overall result across all hits that are part of the same synset, thereby aggregating best
            # match information across multiple potential synonyms
            for hit in hit_set:
                for metric, score in hit.metrics.items():
                    if global_metrics.get(metric, 0) < score:
                        global_metrics[metric] = score
            record.update(global_metrics)
            records.append(record)

        df = pd.DataFrame.from_records(records)
        rank_cols = []
        for colname in df.columns:
            if colname != "syn_data":
                rank_name = f"{colname}_rank"
                rank_cols.append(rank_name)
                df[rank_name] = df[colname].rank(method="min", na_option="bottom", ascending=False)
        # require at least 3 ranks
        if len(rank_cols) >= 3:
            df["total"] = df[rank_cols].sum(axis=1)
            df["total_rank"] = df["total"].rank(method="min", na_option="bottom", ascending=True)
            df.sort_values(by="total_rank", ascending=True, inplace=True)
            hit_found = False
            for i, row in df.iterrows():
                rank_position = row["total_rank"]
                if hit_found or rank_position > self.max_rank_to_consider:
                    break
                syn_data_set = row["syn_data"]
                if self.embedding_field not in row or self.search_field not in row:
                    continue
                search_score = row[self.search_field]
                embedding_score = row[self.embedding_field]
                hit_ok, confidence = self.hit_threshold_condition(
                    search_score=search_score,
                    embedding_score=embedding_score,
                    rank_position=rank_position,
                )
                if hit_ok:
                    for target_syn_data in syn_data_set:
                        if target_syn_data.aggregated_by in UMAMBIGUOUS_SYNONYM_MERGE_STRATEGIES:
                            for idx in target_syn_data.ids:
                                yield DisambiguatedHit(
                                    original_hit=None,
                                    idx=idx,
                                    source=target_syn_data.ids_to_source[idx],
                                    confidence=confidence,
                                    parser_name=hits[0].parser_name,
                                    mapping_type=target_syn_data.mapping_type,
                                )
                                hit_found = True


class RequireHighConfidenceKnowledgeBaseDisambiguationStrategy(KnowledgeBaseDisambiguationStrategy):
    def __call__(
        self, ent_match: str, document: Document, hits: List[Hit]
    ) -> Iterable[DisambiguatedHit]:
        for hit in hits:
            if hit.confidence == SearchRanks.EXACT_MATCH:
                for syn_data in hit.syn_data:
                    if syn_data.aggregated_by in UMAMBIGUOUS_SYNONYM_MERGE_STRATEGIES:
                        for idx in syn_data.ids:
                            yield DisambiguatedHit(
                                original_hit=hit,
                                idx=idx,
                                source=syn_data.ids_to_source[idx],
                                confidence=LinkRanks.MEDIUM_HIGH_CONFIDENCE,
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
        self.hit_post_processor = HitPostProcessor()
        self.synonym_db = SynonymDatabase()
        self.threshold = 7.0

    def prepare(self, document: Document):
        self.query_mat = build_query_matrix_cacheable(document, self.vectoriser, self.manipulator)

    def __call__(
        self, ent_match: str, document: Document, hits: List[Hit]
    ) -> Iterable[DisambiguatedHit]:
        parser_name = hits[0].parser_name
        filtered_hits = filter(lambda x: x.confidence == SearchRanks.EXACT_MATCH, hits)

        filtered_hits = list(filtered_hits)
        syn_data_set_to_hits = {
            k: set(v)
            for k, v in itertools.groupby(
                sorted(filtered_hits, key=lambda x: tuple(x.syn_data)),
                key=lambda x: tuple(x.syn_data),
            )
        }

        synonym_to_syn_data = defaultdict(set)
        for syn_data_set, hits_set in syn_data_set_to_hits.items():
            hit = next(iter(hits_set))
            for syn_data in syn_data_set:
                for idx in syn_data.ids:
                    syns_this_id = self.synonym_db.get_syns_for_id(hit.parser_name, idx)
                    for syn in syns_this_id:
                        synonym_to_syn_data[syn].add(syn_data)

        synonyms_and_scores = list(
            self.corpus_scorer(list(synonym_to_syn_data.keys()), self.query_mat)
        )
        df = pd.DataFrame.from_records(synonyms_and_scores, columns=["synonym", "score"])
        df["rank"] = df["score"].rank(ascending=True, na_option="bottom")

        df = (
            df[["synonym", "score"]]
            .groupby("score")
            .agg(set)
            .reset_index()
            .sort_values("score", ascending=False)
        )

        hit_found = False
        for i, row in df.iterrows():
            score = row["score"]
            if not score > self.threshold:
                break
            #     different syns ay have same score
            synonyms_this_rank = row["synonym"]
            for synonym in synonyms_this_rank:
                syn_data_this_rank = synonym_to_syn_data[synonym]
                # syn is not ambiguous!
                if len(syn_data_this_rank) == 1:
                    syn_data = next(iter(syn_data_this_rank))
                    logger.debug(
                        f'possible hit resolved: original: <{ent_match}> -> <{synonym}>: tfidf: {row["score"]}'
                    )
                    for idx in syn_data.ids:
                        yield DisambiguatedHit(
                            original_hit=None,
                            idx=idx,
                            source=syn_data.ids_to_source[idx],
                            confidence=LinkRanks.MEDIUM_CONFIDENCE,  # set to medium, since we got to this strategy without finding a hit...
                            parser_name=parser_name,
                            mapping_type=syn_data.mapping_type,
                        )
                    hit_found = True
                if hit_found:
                    break
            if hit_found:
                break


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
                    if hit.confidence == SearchRanks.EXACT_MATCH:
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
        self.threshold = 0.5
        # if any entity string matches  are longer than this and have a high confidence hit, assume they're probably right
        self.manipulator = DocumentManipulator()
        self.document_tfidf_matrix = None

    def prepare(self, document: Document):
        self.document_tfidf_matrix = build_query_matrix_cacheable(
            document, self.vectoriser, self.manipulator
        )

    def __call__(
        self, ent_match: str, entities: List[Entity], document: Document
    ) -> Tuple[List[Entity], List[Entity]]:
        """
        get doc representation (full)
        group ents by class
        get all syns per class and build TFIDF matrix
        score
        if syn name is compatible, as before

        :param ent_match:
        :param entities:
        :param document:
        :return:
        """
        class_and_ents = {
            k: set(v)
            for k, v in itertools.groupby(
                sorted(entities, key=lambda x: x.entity_class), key=lambda x: x.entity_class
            )
        }
        # if only one class:
        if len(class_and_ents) == 1:
            return entities, []

        # if either class have high conf hits
        amb, disam = self.sort_by_exact_match(class_and_ents)
        if disam:
            return disam, amb

        else:

            class_to_query = self.entity_class_to_synonym_query(class_and_ents)
            # if no synonyms found
            if len(class_to_query) == 0:
                return [], entities
            elif len(class_to_query) == 1:
                # if only one ent class has synonyms, chose that one
                chosen_class = next(iter(class_to_query.keys()))
                disambiguated_ents = class_and_ents[chosen_class]
                still_ambiguous_ents = pydash.flatten(
                    [ents for clazz, ents in class_and_ents.items() if clazz != chosen_class]
                )
                return list(disambiguated_ents), list(still_ambiguous_ents)
            else:
                # run tfidf on syns
                df = self.build_ranking_dataframe(class_to_query)
                for i, row in df.iterrows():
                    if len(row["class"]) > 1:
                        continue
                    else:
                        chosen_class = next(iter(row["class"]))
                        disambiguated_ents = class_and_ents[chosen_class]
                        still_ambiguous_ents = pydash.flatten(
                            [
                                ents
                                for clazz, ents in class_and_ents.items()
                                if clazz != chosen_class
                            ]
                        )
                        return list(disambiguated_ents), list(still_ambiguous_ents)

                else:
                    return [], entities

    def build_ranking_dataframe(self, class_to_query):
        records = []
        for entity_class, query in class_to_query.items():
            hits_and_scores = list(self.corpus_scorer(list(query), self.document_tfidf_matrix))
            for hit, score in hits_and_scores:
                records.append({"class": entity_class, "hit": hit, "score": score})
        df = pd.DataFrame.from_records(records)
        df["rank"] = df["score"].rank(ascending=True, na_option="bottom")
        df = (
            df[["class", "score"]]
            .groupby("score")
            .agg(set)
            .reset_index()
            .sort_values("score", ascending=False)
        )
        return df

    def entity_class_to_synonym_query(self, class_and_ents):
        class_to_query = {}
        for entity_class, ent_set in class_and_ents.items():
            test_ent = next(iter(ent_set))
            query = set(self.queries.collect_all_syns_from_ents([test_ent]))
            if query:
                class_to_query[entity_class] = query
        return class_to_query

    def sort_by_exact_match(self, class_and_ents):
        disam, amb = [], []
        for entity_class, ent_set in class_and_ents.items():
            test_ent = next(iter(ent_set))
            if any(hit.confidence == SearchRanks.EXACT_MATCH for hit in test_ent.hits):
                disam.extend(list(ent_set))
            else:
                amb.extend(list(ent_set))
        return amb, disam


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

        tfidf_ambiguous_kb_strategy = TfIdfKnowledgeBaseDisambiguationStrategy(self.vectoriser)
        required_full_definition_strategy = (
            RequireFullDefinitionKnowledgeBaseDisambiguationStrategy()
        )
        required_high_confidence = RequireHighConfidenceKnowledgeBaseDisambiguationStrategy()
        prefer_default_label = PreferDefaultLabelKnowledgeBaseDisambiguationStrategy()

        keep_high_conf_global_strategy = KeepHighConfidenceHitsGlobalDisambiguationStrategy()
        tfidf_global_strategy = TfIdfGlobalDisambiguationStrategy(
            self.vectoriser, self.kbs_are_compatible
        )
        hit_ensemble_strategy = HitEnsembleKnowledgeBaseDisambiguationStrategy()

        self.default_strategy = [required_high_confidence]

        self.prefilter_lookup = {
            "gene": [
                prefilter_zero_len,
                prefilter_imprecise_subspans,
                prefilter_unlikely_gene_symbols,
            ],
            "disease": [
                prefilter_zero_len,
                prefilter_imprecise_subspans,
                prefilter_unlikely_acronyms,
            ],
            "drug": [prefilter_zero_len, prefilter_imprecise_subspans, prefilter_unlikely_acronyms],
            "anatomy": [prefilter_zero_len, prefilter_imprecise_subspans],
        }
        self.symbolic_disambiguation_strategy_lookup = {
            "gene": [
                required_full_definition_strategy,
                required_high_confidence,
                prefer_default_label,
                tfidf_ambiguous_kb_strategy,
            ],
            "disease": [
                required_full_definition_strategy,
                required_high_confidence,
                prefer_default_label,
                tfidf_ambiguous_kb_strategy,
            ],
            "drug": [
                required_full_definition_strategy,
                required_high_confidence,
                prefer_default_label,
                tfidf_ambiguous_kb_strategy,
            ],
            "anatomy": [
                required_high_confidence,
                prefer_default_label,
                tfidf_ambiguous_kb_strategy,
            ],
        }
        self.non_symbolic_disambiguation_strategy_lookup = {
            "gene": [
                required_high_confidence,
                prefer_default_label,
                hit_ensemble_strategy,
                tfidf_ambiguous_kb_strategy,
            ],
            "disease": [
                required_high_confidence,
                prefer_default_label,
                hit_ensemble_strategy,
                tfidf_ambiguous_kb_strategy,
            ],
            "drug": [
                required_high_confidence,
                prefer_default_label,
                hit_ensemble_strategy,
                tfidf_ambiguous_kb_strategy,
            ],
            "anatomy": [
                required_high_confidence,
                prefer_default_label,
                hit_ensemble_strategy,
                tfidf_ambiguous_kb_strategy,
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
            tfidf_ambiguous_kb_strategy,
            hit_ensemble_strategy,
            required_full_definition_strategy,
            keep_high_conf_global_strategy,
            tfidf_global_strategy,
            required_full_definition_global,
            required_high_confidence,
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
            logger.debug(
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
                filters = self.prefilter_lookup.get(
                    entity_class,
                    [prefilter_zero_len, prefilter_imprecise_subspans, prefilter_unlikely_acronyms],
                )
                for f in filters:
                    entities_this_class_and_match = f(entities_this_class_and_match)
                # note,assume all ents with same match have same hits!
                if len(entities_this_class_and_match) > 0:
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
                        logger.debug(
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
                            logger.debug(
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


def prefilter_imprecise_subspans(ents: Iterable[Entity]) -> List[Entity]:
    result = []
    metadata_needing_exact_match = ["split_rule", IS_SUBSPAN]
    for ent in ents:
        if any(x in ent.metadata for x in metadata_needing_exact_match):
            if any(hit.confidence == SearchRanks.EXACT_MATCH for hit in ent.hits):
                result.append(ent)
            else:
                logger.debug("filtered %s -> imprecise subspan", ent)
        else:
            result.append(ent)
    return result


def prefilter_zero_len(ents: Iterable[Entity]) -> List[Entity]:
    result = []
    for ent in ents:
        # filter if weird chars in short strings
        if len(ent.match) == 0:
            logger.debug("filtered %s -> zero length!", ent.match)
            break
        else:
            result.append(ent)
    return result


def prefilter_unlikely_acronyms(ents: Iterable[Entity]) -> List[Entity]:
    result = []
    for ent in ents:
        # filter if weird chars in short strings
        if len(ent.match) <= 4:
            for char in ent.match:
                if not (char.isalnum() or char in GreekSymbolSubstitution.GREEK_SUBS):
                    logger.debug("filtered %s -> unlikely acronym", ent.match)
                    break
            else:
                result.append(ent)
        else:
            result.append(ent)
    return result


def prefilter_unlikely_gene_symbols(ents: Iterable[Entity]) -> List[Entity]:
    result = []
    for ent in ents:
        # filter if prefix or suffix looks weird
        if len(ent.match) <= 4:
            if (ent.match[0].isalnum() or ent.match[0] in GreekSymbolSubstitution.GREEK_SUBS) and (
                ent.match[-1].isalnum() or ent.match[-1] in GreekSymbolSubstitution.GREEK_SUBS
            ):
                result.append(ent)
            else:
                logger.debug("filtered %s -> unlikely gene symbol", ent.match)
        else:
            result.append(ent)
    return result


class DocumentLevelDisambiguationStep(BaseStep):

    """
    algorithm:

    there are two scenarios:

    a) entities with better than low confidence mappings, but are ambiguous
    b) entities with only low confidence mappings that are unambiguous

    """

    def __init__(self, depends_on: Optional[List[str]], tfidf_disambiguator: Disambiguator):
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
