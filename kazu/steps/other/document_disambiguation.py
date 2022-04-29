import copy
import itertools
import logging
import pickle
import re
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set, FrozenSet, KeysView, Iterable, Callable

import numpy as np
import pandas as pd
import pydash
import torch
from kazu.data.data import (
    Document,
    Mapping,
    LINK_SCORE,
    NAMESPACE,
    Entity,
    LINK_CONFIDENCE,
    SynonymData,
)
from kazu.data.data import LinkRanks
from kazu.modelling.ontology_preprocessing.base import (
    DEFAULT_LABEL,
    MAPPING_TYPE,
    MetadataDatabase,
    SynonymDatabase,
    StringNormalizer,
)
from kazu.steps import BaseStep
from kazu.utils.link_index import Hit, DICTIONARY_HITS, create_char_ngrams, SAPBERT_SCORE
from kazu.utils.utils import HitResolver, ParseSourceFromId
from sklearn.feature_extraction.text import TfidfVectorizer
from strsimpy import NGram, LongestCommonSubsequence

logger = logging.getLogger(__name__)

DISAMBIGUATED_BY = "disambiguated_by"
DISAMBIGUATED_BY_DEFINED_ELSEWHERE = "defined elsewhere in document"
DISAMBIGUATED_BY_REACTOME = "reactome pathway links in document"
DISAMBIGUATED_BY_CONTEXT = "document context"
KB_DISAMBIGUATION_FAILURE = "unable_to_disambiguate_within_ontology"
GLOBAL_DISAMBIGUATION_FAILURE = "unable_to_disambiguate_on_context"
SUGGEST_DROP = "suggest_drop"


class SynonymDataDisambiguationStrategy:
    def __init__(self, entity_match_string: str, check_for_synonym_string_match: bool = False):
        self.entity_match_string = entity_match_string
        self.ent_match_norm = StringNormalizer.normalize(entity_match_string)
        self.number_resolver = NumberResolver(self.ent_match_norm)
        self.string_resolver = (
            SubStringResolver(self.ent_match_norm) if check_for_synonym_string_match else None
        )
        self.synonym_db = SynonymDatabase()
        self.metadata_db = MetadataDatabase()
        self.minimum_string_length_for_non_exact_mapping = 4

    def resolve_short_strings(self, ambig_hits: List[Hit]) -> List[Hit]:
        if len(self.entity_match_string) < self.minimum_string_length_for_non_exact_mapping:
            return list(filter(lambda x: x.confidence == LinkRanks.HIGH_CONFIDENCE, ambig_hits))
        else:
            return ambig_hits

    def resolve_synonym_and_source(self, synonym: str, source: str) -> Optional[SynonymData]:
        if not self.number_resolver(synonym):
            logger.debug(f"{synonym} still ambiguous: number mismatch: {self.ent_match_norm}")
            return None
        if self.string_resolver and not self.string_resolver(synonym):
            logger.debug(f"{synonym} still ambiguous: substring not found: {self.ent_match_norm}")
            return None

        syn_data_set_this_hit: Set[SynonymData] = self.synonym_db.get(name=source, synonym=synonym)
        target_syn_data = None
        if len(syn_data_set_this_hit) > 1:
            # if synonym is a short string, continue
            # if synonym is long, chances are we can get a match with Sapbert
            if len(synonym) < 5:
                logger.debug(f"{synonym} still ambiguous: {syn_data_set_this_hit}")
            else:
                logger.debug(
                    f"{synonym} is ambiguous, but string is long. Attempting ngram disambiguation"
                )
                target_syn_data = self.ngram_disambiguation(
                    source=source,
                    synonym=synonym,
                    syn_data_set_this_hit=syn_data_set_this_hit,
                )
        elif len(syn_data_set_this_hit) == 1:
            target_syn_data = next(iter(syn_data_set_this_hit))
        return target_syn_data

    def ngram_disambiguation(
        self, source: str, synonym: str, syn_data_set_this_hit: Set[SynonymData]
    ) -> SynonymData:
        """
        may be replaced with Sapbert
        :param source:
        :param synonym:
        :param syn_data_set_this_hit:
        :return:
        """
        # TODO: needs threshold
        ngram = NGram(2)
        idx_and_default_labels = []
        for syn_data in syn_data_set_this_hit:
            for idx in syn_data.ids:
                metadata = self.metadata_db.get_by_idx(name=source, idx=idx)
                idx_and_default_labels.append(
                    (
                        syn_data,
                        StringNormalizer.normalize(metadata[DEFAULT_LABEL]),
                    )
                )
        scores = []
        for syn_data, default_label in idx_and_default_labels:
            score = ngram.distance(synonym, default_label)
            scores.append(
                (
                    syn_data,
                    default_label,
                    score,
                )
            )
        result = sorted(scores, key=lambda x: x[1], reverse=False)[0]
        logger.debug(f"ngram disambiguated {synonym} to {result[1]} with score: {result[2]}")
        return result[0]


class SynonymDbQueryExtensions:
    def __init__(self):
        self.synonym_db = SynonymDatabase()

    def create_corpus_for_source(self, ambig_hits_this_source: List[Hit], source) -> List[str]:
        ambig_ids_this_source = {
            idx
            for hit in ambig_hits_this_source
            for syn_data in hit.syn_data
            for idx in syn_data.ids
        }
        # build the corpus for all hits in this list
        corpus = []
        for idx in ambig_ids_this_source:
            for syn in self.synonym_db.get_syns_for_id(name=source, idx=idx):
                corpus.append(syn)
        return corpus

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
                                    list(self.synonym_db.get_syns_for_id(name=hit.source, idx=idx))
                                )
                            )
        return result


class OntologyDisambiguationStrategies(Enum):
    CONTEXT_TFIDF = "context_tfidf"
    CONTEXT_SAPBERT = "context_sapbert"
    ALL_SUBSTRING = "all_substring"


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


class TfIdfOntologyHitDisambiguationStrategy:
    def __init__(self, vectoriser: TfidfVectorizer, id_parser: ParseSourceFromId):
        self.id_parser = id_parser
        self.metadata_db = MetadataDatabase()
        self.vectoriser = vectoriser
        self.corpus_scorer = TfIdfCorpusScorer(vectoriser)
        self.queries = SynonymDbQueryExtensions()

    def get_query_mat(self, document_representation: List[str]):
        query = " . ".join(document_representation)
        query_mat = self.vectoriser.transform([query]).todense()
        return query_mat

    def get_synonym_data_disambiguating_strategy(self, entity_string: str):
        return SynonymDataDisambiguationStrategy(entity_string)

    def __call__(
        self, entity_string: str, entities: List[Entity], document_representation: List[str]
    ):
        query = " . ".join(document_representation)
        query_mat = self.vectoriser.transform([query]).todense()
        disambuguated_mappings = []
        synonym_data_disambiguator = self.get_synonym_data_disambiguating_strategy(entity_string)
        ambig_hits = {hit for ent in entities for hit in ent.hits}

        # TODO: optimisation: don't repeat loop if score is the same as prev (i.e. is same string)
        hits_by_source = itertools.groupby(
            sorted(ambig_hits, key=lambda x: x.source), key=lambda x: x.source
        )
        for source, hits_iter in hits_by_source:
            # the correct id is most likely in this list
            ambig_hits_this_source = list(hits_iter)
            ambig_hits_this_source = synonym_data_disambiguator.resolve_short_strings(
                ambig_hits_this_source
            )

            corpus = self.queries.create_corpus_for_source(ambig_hits_this_source, source)

            syns_and_scores = list(self.corpus_scorer(corpus, query_mat))
            syns_and_scores = sorted(syns_and_scores, key=lambda x: x[1], reverse=True)

            for synonym, score in syns_and_scores:
                target_syn_data = synonym_data_disambiguator.resolve_synonym_and_source(
                    source=source, synonym=synonym
                )
                if target_syn_data:
                    for idx in target_syn_data.ids:
                        metadata = self.metadata_db.get_by_idx(name=source, idx=idx)
                        metadata[DISAMBIGUATED_BY] = DISAMBIGUATED_BY_CONTEXT
                        metadata[LINK_CONFIDENCE] = LinkRanks.MEDIUM_HIGH_CONFIDENCE
                        disambuguated_mappings.append(
                            Mapping(
                                default_label=metadata[DEFAULT_LABEL],
                                idx=idx,
                                source=self.id_parser(source, idx),
                                mapping_type=target_syn_data.mapping_type,
                                metadata=metadata,
                            )
                        )
                        logger.debug(f"{synonym} disambiguated. id: {idx}, score: {score}")
                    break
            else:
                logger.debug(
                    f"failed to disambiguate {entity_string}. Ambiguous hits {ambig_hits_this_source}"
                )
            for ent in entities:
                ent.mappings.extend(copy.deepcopy(disambuguated_mappings))


class TfIdfOntologyHitDisambiguationStrategyWithSubStringMatching(
    TfIdfOntologyHitDisambiguationStrategy
):
    """
    same as superclass, but checks matched entity is a substring of synonym
    """

    def get_synonym_data_disambiguating_strategy(self, entity_string: str):
        return SynonymDataDisambiguationStrategy(entity_string, check_for_synonym_string_match=True)


class TfIdfGlobalDisambiguationStrategy:
    def __init__(self, vectoriser: TfidfVectorizer, kbs_are_compatible: Callable[[Set[str]], bool]):
        self.kbs_are_compatible = kbs_are_compatible
        self.queries = SynonymDbQueryExtensions()
        self.vectoriser = vectoriser
        self.corpus_scorer = TfIdfCorpusScorer(vectoriser)
        self.threshold = 7.0
        # if any entity string matches  are longer than this and have a high confidence hit, assume they're probably right

        self.min_string_length_to_test_for_high_confidence = 5

    def __call__(self, entities: List[Entity], document_representation: List[str]) -> List[Entity]:
        result = []
        query = " . ".join(document_representation)
        query_mat = self.vectoriser.transform([query]).todense()

        ents_by_match = itertools.groupby(
            sorted(entities, key=lambda x: x.match), key=lambda x: x.match
        )
        for match_str, ent_iter in ents_by_match:
            resolved_ents_this_match = []
            match_ents = list(ent_iter)

            if len(match_str) >= self.min_string_length_to_test_for_high_confidence:
                for ent in match_ents:
                    if any(hit.confidence == LinkRanks.HIGH_CONFIDENCE for hit in ent.hits):
                        resolved_ents_this_match.append(ent)

            # we found high confidence hits, so don't go any further
            if resolved_ents_this_match:
                result.extend(resolved_ents_this_match)
                continue

            # group ents by hit source
            hit_sources_to_ents = defaultdict(set)
            for ent in match_ents:
                for hit in ent.hits:
                    hit_sources_to_ents[hit.source].add(ent)

            # otherwise, test context for good hits...
            corpus = list(set(self.queries.collect_all_syns_from_ents(match_ents)))
            for synonym, score in self.corpus_scorer(corpus, query_mat):
                if score < self.threshold:
                    # no ents could be resolved this match
                    for ent in match_ents:
                        ent.metadata[SUGGEST_DROP] = GLOBAL_DISAMBIGUATION_FAILURE
                    break
                kbs_this_hit = self.queries.synonym_db.get_kbs_for_syn_global(synonym)
                if not self.kbs_are_compatible(kbs_this_hit):
                    logger.debug(f"class still ambiguous: {kbs_this_hit}")
                else:
                    # for the first unambiguous kb hit,return the ent that has a hit with this kb.
                    for kb_hit in kbs_this_hit:
                        ents_this_kb = hit_sources_to_ents.get(kb_hit)
                        if ents_this_kb:
                            resolved_ents_this_match.extend(list(ents_this_kb))
                            break
                    # we found a match via context
                if resolved_ents_this_match:
                    result.extend(resolved_ents_this_match)
                    break

        return result


def create_char_3grams(string):
    return create_char_ngrams(string, n=3)


class Disambiguator:
    """
    needs to
    a) discover ents needing_global_disambiguation, and disambiguate-> discover entity class from Explosion
    b) resolve any 'clean' hits (via HitResolver)
    c) group remaining ents by string
    d) group further into ent class
    e) select disambig strategy ranker based on entity class -> Set SynonymData
    f) for resulting ranks,
    f) if len(syndata)>1, decide whether to attempt disambiguation within syndata set
    g)


    """

    def __init__(self, path: str, id_parser: ParseSourceFromId):
        self.id_parser = id_parser
        self.vectoriser: TfidfVectorizer = self.build_or_load_vectoriser(path)
        self.syn_db = SynonymDatabase()
        self.metadata_db = MetadataDatabase()
        self.allowed_overlaps = {
            frozenset({"MONDO", "MEDDRA", "OPENTARGETS_DISEASE"}),
            frozenset({"CHEMBL", "OPENTARGETS_MOLECULE", "OPENTARGETS_TARGET"}),
        }
        self.always_disambiguate = {"ExplosionNERStep"}
        self.hit_resolver = HitResolver(id_parser)

        self.context_tfidf = TfIdfOntologyHitDisambiguationStrategy(
            vectoriser=self.vectoriser, id_parser=id_parser
        )
        self.context_tfidf_with_string_matching = (
            TfIdfOntologyHitDisambiguationStrategyWithSubStringMatching(
                vectoriser=self.vectoriser, id_parser=id_parser
            )
        )

        self.disambiguation_strategy_lookup = {
            "gene": self.context_tfidf,
            "disease": self.context_tfidf,
            "drug": self.context_tfidf_with_string_matching,
        }
        self.global_disambiguation_strategy = TfIdfGlobalDisambiguationStrategy(
            vectoriser=self.vectoriser, kbs_are_compatible=self.kbs_are_compatible
        )

    def build_or_load_vectoriser(self, path_str: str):
        path = Path(path_str)
        if path.exists():
            return pickle.load(open(path, "rb"))
        else:
            vec = TfidfVectorizer(lowercase=False, analyzer=create_char_3grams)
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

    def get_document_representation(self, entities: List[Entity]) -> List[str]:
        return [StringNormalizer.normalize(x.match) for x in entities]

    def run(self, doc: Document):
        entities = doc.get_entities()
        document_representation = self.get_document_representation(entities)
        (
            entities_needing_global_disamb,
            entities_not_needing_global_disamb,
        ) = self.find_entities_needing_global_disambiguation(entities)

        globally_disambiguated_ents = self.global_disambiguation_strategy(
            entities_needing_global_disamb, document_representation
        )

        remaining_ents = self.resolve_unambiguous_entities(
            entities_not_needing_global_disamb, globally_disambiguated_ents
        )

        # TODO: add flag to run sapbert on anything that fails to disambig within kb
        self.disambiguate_within_kb(remaining_ents, document_representation)

    def find_entities_needing_global_disambiguation(self, entities):
        entities_needing_global_disamb = []
        entities_not_needing_global_disamb = []
        for ent in entities:
            if ent.namespace in self.always_disambiguate:
                entities_needing_global_disamb.append(ent)
            else:
                entities_not_needing_global_disamb.append(ent)
        return entities_needing_global_disamb, entities_not_needing_global_disamb

    def resolve_unambiguous_entities(
        self,
        entities_not_needing_global_disamb: List[Entity],
        globally_disambiguated_ents: List[Entity],
    ):
        remaining_ents = entities_not_needing_global_disamb + globally_disambiguated_ents
        # the remainng ents should now be 'good' in terms of ner entity_class and therefore we know which kbs
        # to disambiguate each of them to. We can now call the HitResolver to turn any unambiguous and betetr than
        # linkranks.low_conf to convert them into mappings
        for ent in list(remaining_ents):
            for hit in list(ent.hits):
                if hit.confidence != LinkRanks.LOW_CONFIDENCE:
                    mappings = list(self.hit_resolver(hit))
                    if mappings:
                        ent.mappings.extend(mappings)
                        # since the hit is now resolved, we no longer need the original hit
            # if the entity now has a good mapping, we don't need to disambiguate further
            if ent.mappings:
                remaining_ents.remove(ent)
        return remaining_ents

    def disambiguate_within_kb(
        self,
        ents_needing_disambig: List[Entity],
        document_representation: List[str],
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

        # clean_ids = {mapping.idx for ent in unambiguoys_ents for mapping in ent.mappings}

        grouped_by_match = itertools.groupby(
            sorted(
                ents_needing_disambig,
                key=lambda x: (
                    x.match,
                    x.entity_class,
                ),
            ),
            key=lambda x: (
                x.match,
                x.entity_class,
            ),
        )

        for ent_match_and_class, ent_iter in grouped_by_match:
            ent_match = ent_match_and_class[0]
            ent_class = ent_match_and_class[1]
            ents_this_match = list(ent_iter)
            self.disambiguation_strategy_lookup.get(ent_class, self.context_tfidf)(
                ent_match, ents_this_match, document_representation
            )

    def score_hits_by_sapbert_calc_syns(self, ent_match: str, ents_this_match: List[Entity]):
        disambuguated_mappings = []
        ent_match_norm = StringNormalizer.normalize(ent_match)
        number_resolver = NumberResolver(ent_match_norm)
        ambig_hits = {hit for ent in ents_this_match for hit in ent.hits}
        hits_by_source = itertools.groupby(
            sorted(ambig_hits, key=lambda x: x.source), key=lambda x: x.source
        )
        query = self.model.get_embeddings_for_strings([ent_match_norm])
        for source, hits_iter in hits_by_source:
            # the correct id is most likely in this list
            ambig_hits_this_source = list(hits_iter)
            corpus = set(self.create_corpus_for_source(ambig_hits_this_source, source))
            syns_and_scores = list(self.score_corpus_sapbert(query, corpus))
            for synonym, score in syns_and_scores:
                # check number match for hit
                if not number_resolver(synonym):
                    continue
                syn_data_set_this_hit = self.syn_db.get(name=source, synonym=synonym)
                if len(syn_data_set_this_hit) == 1:
                    target_syn_data = next(iter(syn_data_set_this_hit))
                    if target_syn_data:
                        for idx in target_syn_data.ids:
                            metadata = self.metadata_db.get_by_idx(name=source, idx=idx)
                            metadata[DISAMBIGUATED_BY] = "sapbert_context"
                            metadata[LINK_CONFIDENCE] = LinkRanks.MEDIUM_HIGH_CONFIDENCE
                            disambuguated_mappings.append(
                                Mapping(
                                    default_label=metadata[DEFAULT_LABEL],
                                    idx=idx,
                                    source=source,
                                    mapping_type=target_syn_data.mapping_type,
                                    metadata=metadata,
                                )
                            )
                            logger.debug(f"{synonym} disambiguated. id: {idx}, score: {score}")
                        break

            else:
                logger.debug(
                    f"failed to disambiguate {ent_match}. Ambiguous hits {ambig_hits_this_source}"
                )
            for ent in ents_this_match:
                ent.mappings.extend(copy.deepcopy(disambuguated_mappings))

    def score_hits_by_sapbert(self, ent_match: str, ents_this_match: List[Entity]):
        disambuguated_mappings = []
        ent_match_norm = StringNormalizer.normalize(ent_match)
        number_resolver = NumberResolver(ent_match_norm)
        ambig_hits = {
            hit
            for ent in ents_this_match
            for hit in ent.hits
            if hit.namespace == "SapBertForEntityLinkingStep"
        }
        hits_by_source = itertools.groupby(
            sorted(ambig_hits, key=lambda x: x.source), key=lambda x: x.source
        )
        for source, hits_iter in hits_by_source:
            # the correct id is most likely in this list
            ambig_hits_this_source = sorted(
                list(hits_iter), key=lambda x: x.metrics[SAPBERT_SCORE], reverse=True
            )

            for hit in ambig_hits_this_source:
                synonym = hit.matched_str
                score = hit.metrics[SAPBERT_SCORE]
                if not number_resolver(synonym):
                    continue
                syn_data_set_this_hit = self.syn_db.get(name=source, synonym=synonym)
                if len(syn_data_set_this_hit) == 1:
                    target_syn_data = next(iter(syn_data_set_this_hit))
                    if target_syn_data:
                        for idx in target_syn_data.ids:
                            metadata = self.metadata_db.get_by_idx(name=source, idx=idx)
                            metadata[DISAMBIGUATED_BY] = "sapbert_context"
                            metadata[LINK_CONFIDENCE] = LinkRanks.MEDIUM_HIGH_CONFIDENCE
                            disambuguated_mappings.append(
                                Mapping(
                                    default_label=metadata[DEFAULT_LABEL],
                                    idx=idx,
                                    source=source,
                                    mapping_type=target_syn_data.mapping_type,
                                    metadata=metadata,
                                )
                            )
                            logger.debug(f"{synonym} disambiguated. id: {idx}, score: {score}")
                        break

            else:
                logger.debug(
                    f"failed to disambiguate {ent_match}. Ambiguous hits {ambig_hits_this_source}"
                )
            for ent in ents_this_match:
                ent.mappings.extend(copy.deepcopy(disambuguated_mappings))

    def score_corpus_sapbert(self, query, corpus: List[str]) -> Iterable[Tuple[str, float]]:
        # todo: implement caching
        # corpus_dict = {}
        # needs_embeddings = []
        # for i, corpus_str in corpus:
        #     cached_embeddings = self.embedding_cache.get(corpus_str)
        #     if cached_embeddings:
        #         corpus_dict[i] = corpus_str
        #     else:
        #         corpus_dict[i] - self.model.get_embeddings_for_strings()
        #
        corpus = list(corpus)
        embeddings = self.model.get_embeddings_for_strings(
            corpus, batch_size=len(corpus), trainer=self.trainer
        )
        score_matrix = torch.matmul(query, embeddings.T)
        score_matrix = torch.squeeze(score_matrix)
        neighbours = torch.argsort(score_matrix, descending=True)
        distances = score_matrix[neighbours]
        distances = 100 - (1 / distances)
        for score, n in zip(distances, neighbours):
            yield corpus[n], score


class ReactomeDb:

    # reactome:
    # Source database identifier, e.g. UniProt, ENSEMBL, NCBI Gene or ChEBI identifier
    # Reactome Pathway Stable identifier
    # URL
    # Event (Pathway or Reaction) Name
    # Evidence Code
    # Species

    def __init__(self, idx_to_pathways: Dict[str, Set[int]], pathways_to_idx: Dict[int, Set[str]]):
        self.pathways_to_idx = pathways_to_idx
        self.idx_to_pathways = idx_to_pathways

    def get_ids_by_pathway_association(self, query: str) -> List[Set[str]]:
        pathways: Set[int] = self.idx_to_pathways.get(query, set())
        results = []
        for pathway in pathways:
            results.append(self.pathways_to_idx[pathway])
        return results

    def rank_pathways(
        self,
        document_unambiguous_ids: KeysView[Tuple[str, str, LinkRanks]],
        query: FrozenSet[SynonymData],
    ) -> Optional[str]:
        ranks = []

        for ambiguous_syn_data in query:
            for idx in ambiguous_syn_data.ids:
                ranks.append(
                    (
                        self.score_pathways(
                            document_unambiguous_ids=document_unambiguous_ids,
                            ambiguous_id=idx,
                        ),
                        idx,
                    ),
                )
        ranks.sort(key=lambda x: x[0], reverse=True)
        top_score, disambiguated_id = ranks[0]
        if top_score > 0.0:
            return disambiguated_id
        else:
            return None

    def score_pathways(
        self, document_unambiguous_ids: KeysView[Tuple[str, str, LinkRanks]], ambiguous_id: str
    ) -> float:
        pathway_sets = sorted(self.get_ids_by_pathway_association(ambiguous_id), reverse=True)
        score = 0.0
        # always prefer confidence values over scores, and return early if a higher confidence result found
        document_unambiguous_ids_by_confidence = itertools.groupby(
            sorted(document_unambiguous_ids, key=lambda x: x[2]), key=lambda x: x[2]
        )

        for link_rank, link_rank_unambiguous in document_unambiguous_ids_by_confidence:
            unambig_ids_this_rank = {x[1] for x in link_rank_unambiguous}
            for pathway in pathway_sets:
                overlap_this_pathway = len(pathway.intersection(unambig_ids_this_rank))
                pathway_score = overlap_this_pathway / len(pathway)
                if pathway_score > score:
                    score = pathway_score
            if score > 0.0:
                break
        return score


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

    to tackle a)
    1. get all document mappings as a list of ambig and unambig, that are better than low conf
    2. map both into dict by {(source,idx,conf):Mapping}
    3. see if any ambig keys in unambig. If so, this suggests the entities with ambig ones should have the unambig ones

    4. group by {(Entity,source,idx,conf):List[Entity]}
    5. loop over
    k,v for k,v in x.items()
    mapping_lookup = tuple(k[1],k[2])
    if mapping_lookup in unambig
     ent = k[0]
     new_mapping = copy.deepcopy(unambig[mapping_lookup])
     ent.mappings = [new_mapping]


    if not, query reactome

    to tackle b)

    repeat a), but only for ents that have only low confidence mappings










    """

    def __init__(
        self,
        depends_on: Optional[List[str]],
        kb_disambiguation_map: Dict[str, str],
        tfidf_disambiguator: Disambiguator,
    ):
        """

        :param depends_on:
        :param namespace_preferred_order: order of namespaces to prefer. Any exactly overlapped entities are eliminated
        according to this ordering (first = higher priority).
        """

        super().__init__(depends_on)
        self.tfidf_disambiguator = tfidf_disambiguator
        self.kb_disambiguation_map = {
            source: self.load_kb(path) for source, path in kb_disambiguation_map.items()
        }
        self.metadata_db = MetadataDatabase()
        self.synonym_db = SynonymDatabase()

    def load_kb(self, path: str) -> ReactomeDb:
        df = pd.read_csv(path, sep="\t")
        df.columns = ["idx", "pathway", "url", "event", "evidence", "species"]
        df = df[["idx", "pathway"]]
        # save some mem by converting to codes
        df["pathway"] = pd.Categorical(df["pathway"]).codes
        df_idx = df.groupby(by="idx").agg(set)
        df_pathway = df.groupby(by="pathway").agg(set)
        return ReactomeDb(df_idx.to_dict()["pathway"], df_pathway.to_dict()["idx"])

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs: List[Document] = []
        for doc in docs:
            self.tfidf_disambiguator.run(doc)
            # self.disambiguate(doc.get_entities())
        return docs, failed_docs

    def find_ambiguous_mappings(
        self, entities: List[Entity]
    ) -> Tuple[
        Dict[Tuple[Entity, str, FrozenSet[SynonymData]], Mapping],
        Dict[Tuple[str, str, LinkRanks], Mapping],
    ]:
        """
        TODO - if linker A provides a high confidence match, whereas linker B provides a low confidence match on the
        same string, what should the result be?
        :param entities:
        :return:
        """
        ambig, unambig = {}, {}
        for ent in entities:
            for mapping in ent.mappings:
                ambig_syns_list = []
                for hit in mapping.metadata.get(DICTIONARY_HITS, []):
                    for syn_data in hit.syn_data:
                        ambig_syns_list.append(syn_data)

                ambig_syns: FrozenSet[SynonymData] = frozenset(ambig_syns_list)
                link_confidence: LinkRanks = mapping.metadata.get(
                    LINK_CONFIDENCE, LinkRanks.LOW_CONFIDENCE
                )
                if not ambig_syns and (
                    not link_confidence == LinkRanks.AMBIGUOUS
                    or not link_confidence == LinkRanks.LOW_CONFIDENCE
                ):
                    unambig[
                        (
                            mapping.source,
                            mapping.idx,
                            link_confidence,
                        )
                    ] = mapping
                else:
                    if ambig_syns:
                        ambig[(ent, mapping.source, frozenset(ambig_syns))] = mapping
                    else:
                        ambig[
                            (
                                ent,
                                mapping.source,
                                frozenset(
                                    [
                                        SynonymData(
                                            ids=frozenset([mapping.idx]),
                                            mapping_type=mapping.mapping_type,
                                        )
                                    ]
                                ),
                            )
                        ] = mapping
        return ambig, unambig

    def disambiguate_by_knowledge_base(
        self,
        good_ids: KeysView[Tuple[str, str, LinkRanks]],
        to_disambiguate: Tuple[Entity, str, FrozenSet[SynonymData]],
    ) -> bool:
        source = to_disambiguate[1]

        if source in self.kb_disambiguation_map:
            syn_data = to_disambiguate[2]
            # if there is pathway info available, search the pathway db for how they might be connected
            result = self.kb_disambiguation_map[source].rank_pathways(
                document_unambiguous_ids=good_ids,
                query=syn_data,
            )
            if result:
                metadata = self.metadata_db.get_by_idx(source, result)
                default_label = metadata.pop(DEFAULT_LABEL, "na")
                mapping_type = frozenset([str(x) for x in metadata.pop(MAPPING_TYPE, ["na"])])
                metadata[LINK_SCORE] = 100.0
                metadata[LINK_CONFIDENCE] = LinkRanks.MEDIUM_HIGH_CONFIDENCE
                metadata[NAMESPACE] = self.namespace()
                metadata[DISAMBIGUATED_BY] = DISAMBIGUATED_BY_REACTOME
                new_good_mapping = Mapping(
                    default_label=str(default_label),
                    source=source,
                    idx=result,
                    mapping_type=mapping_type,
                    metadata=metadata,
                )
                ent = to_disambiguate[0]
                ent.mappings = [new_good_mapping]
                return True
        return False

    def disambiguate(self, entities: List[Entity]):
        dicts = self.find_ambiguous_mappings(entities)
        ambig: Dict[Tuple[Entity, str, FrozenSet[SynonymData]], Mapping] = dicts[0]
        unambig: Dict[Tuple[str, FrozenSet[str], LinkRanks], Mapping] = dicts[1]

        unresolved_ambig_ents = []
        for key, mapping in ambig.items():
            ent: Entity = key[0]
            source: str = key[1]
            ambig_syns: FrozenSet[SynonymData] = key[2]
            for ambig_syn in ambig_syns:
                for link_rank in LinkRanks:
                    unambig_lookup: Tuple[str, FrozenSet[str], LinkRanks] = tuple(
                        [source, ambig_syn.ids, link_rank]
                    )
                    preferred_mapping = unambig.get(unambig_lookup)
                    if preferred_mapping:
                        new_mapping = copy.deepcopy(preferred_mapping)
                        new_mapping.metadata[DISAMBIGUATED_BY] = DISAMBIGUATED_BY_DEFINED_ELSEWHERE
                        ent.mappings = [new_mapping]
                        break
                else:
                    continue
                break
            else:
                if not self.disambiguate_by_knowledge_base(
                    good_ids=unambig.keys(), to_disambiguate=key
                ):
                    unresolved_ambig_ents.append(ent)
        return unresolved_ambig_ents
