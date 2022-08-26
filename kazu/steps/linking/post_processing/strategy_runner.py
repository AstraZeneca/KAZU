import logging
from typing import List, Tuple, Iterable, Dict, FrozenSet

from kazu.data.data import (
    Document,
    Entity,
    SynonymTermWithMetrics,
)
from kazu.modelling.database.in_memory_db import MetadataDatabase
from kazu.steps.linking.post_processing.string_matching.strategies import StringMatchingStrategy
from kazu.utils.grouping import sort_then_group
from kazu.utils.string_normalizer import StringNormalizer

logger = logging.getLogger(__name__)


EntityClassStrategy = Dict[str, List[StringMatchingStrategy]]


class NamespaceStrategyList:
    def __init__(
        self,
        ent_class_strategies: EntityClassStrategy,
        default_strategies: List[StringMatchingStrategy],
    ):
        self.default_strategies = default_strategies
        self.ent_class_strategies = ent_class_strategies

    def get_strategies_for_entity_class(self, entity_class: str) -> List[StringMatchingStrategy]:
        return self.ent_class_strategies.get(entity_class, self.default_strategies)


class StrategyRunner:
    def __init__(
        self,
        symbolic_strategies: Dict[str, NamespaceStrategyList],
        non_symbolic_strategies: Dict[str, NamespaceStrategyList],
        ner_namespace_processing_order: List[str],
    ):
        """

        :param non_symbolic_strategies:
        :param symbolic_strategies:
        :param symbol_classifier_lookup:
        :param ner_namespace_processing_order: Entities will be mapped in this namespace order. This is
            useful if you have a high precision, low recall NER namespace, combined with a low precision high recall
            namespace, as the mapping info derived from the high precision NER namespace can be used with a high
            precision strategy for the low precision NER namespace
        """
        self.non_symbolic_strategies = non_symbolic_strategies
        self.symbolic_strategies = symbolic_strategies
        self.ner_namespace_processing_order = ner_namespace_processing_order
        self.metadata_db = MetadataDatabase()

    def sort_entities_by_symbolism(
        self, entities: Iterable[Entity]
    ) -> Tuple[List[Entity], List[Entity]]:
        """
        sorts entities into symbolic and non-symbolic forms, so they can be processed separately
        :param entities:
        :return:
        """
        symbolic, non_symbolic = [], []
        grouped_by_match = sort_then_group(
            entities,
            key_func=lambda x: (
                x.match,
                x.entity_class,
            ),
        )
        for (match_str, entity_class), ent_iter in grouped_by_match:
            if StringNormalizer.classify_symbolic(match_str, entity_class=entity_class):
                symbolic.extend(list(ent_iter))
            else:
                non_symbolic.extend(list(ent_iter))
        return symbolic, non_symbolic

    def __call__(self, doc: Document):

        if self.ner_namespace_processing_order is None:
            entities_grouped_by_namespace_order = sort_then_group(
                doc.get_entities(), key_func=lambda x: x.namespace
            )
        else:
            entities_grouped_by_namespace_order = sort_then_group(
                doc.get_entities(),
                key_func=lambda x: self.ner_namespace_processing_order.index(x.namespace),
            )

        for namespace_index, entities in entities_grouped_by_namespace_order:
            namespace = self.ner_namespace_processing_order[namespace_index]
            logger.debug("mapping entities for namespace %s", namespace)
            symbolic_entities, non_symbolic_entities = self.sort_entities_by_symbolism(entities)
            self.execute_hit_post_processing_strategies(
                non_symbolic_entities, doc, self.non_symbolic_strategies[namespace]
            )
            self.execute_hit_post_processing_strategies(
                symbolic_entities, doc, self.symbolic_strategies[namespace]
            )

    def execute_hit_post_processing_strategies(
        self,
        ents_needing_mappings: List[Entity],
        document: Document,
        namespace_strategy_list: NamespaceStrategyList,
    ):
        """


        :param entitites:
        :param query_mat:
        :return:
        """

        strategy_max_index = max(
            [
                len(strategies)
                for strategies in namespace_strategy_list.ent_class_strategies.values()
            ]
            + [len(namespace_strategy_list.default_strategies)]
        )

        def _get_key_to_group_ent_on_hits(
            e: Entity,
        ) -> Tuple[str, str, str, FrozenSet[SynonymTermWithMetrics]]:
            return (
                e.match,
                e.match_norm,
                e.entity_class,
                frozenset(e.syn_term_to_synonym_terms.values()),
            )

        groups = {
            k: list(v)
            for k, v in sort_then_group(ents_needing_mappings, _get_key_to_group_ent_on_hits)
        }

        for i in range(0, strategy_max_index):
            for (
                entity_match,
                entity_match_norm,
                entity_class,
                terms,
            ), entities_this_group in groups.items():

                strategy_list: List[
                    StringMatchingStrategy
                ] = namespace_strategy_list.get_strategies_for_entity_class(entity_class)
                if i > len(strategy_list) - 1:
                    logger.debug("no more strategies this class")
                    continue
                else:
                    strategy = strategy_list[i]
                    logger.debug(
                        f"running strategy {strategy.__class__.__name__} on class :<{entity_class}>, match: <{entity_match}> "
                    )
                    strategy.prepare(document)

                    # we keep track of which entities have resolved mappings to specific parsers, so we don't run lower
                    # ranked strategies if we don't need to

                    query_entity = next(iter(entities_this_group))
                    resolved_parsers = set(mapping.parser_name for mapping in query_entity.mappings)
                    terms_to_consider = filter(
                        lambda x: x.parser_name not in resolved_parsers,
                        terms,
                    )
                    terms_by_parser = sort_then_group(
                        terms_to_consider, key_func=lambda x: x.parser_name
                    )

                    for parser_name, hits_this_parser in terms_by_parser:
                        terms_this_parser_set = frozenset(hits_this_parser)
                        if len(terms_this_parser_set) == 0:
                            continue
                        for mapping in strategy(
                            ent_match=entity_match,
                            ent_match_norm=entity_match_norm,
                            terms=terms_this_parser_set,
                            document=document,
                        ):
                            # add mappings to the entity
                            for entity in entities_this_group:
                                entity.mappings.add(mapping)
                            logger.debug(
                                "mapping created: original string: %s, mapping: %s",
                                entity_match,
                                mapping,
                            )
