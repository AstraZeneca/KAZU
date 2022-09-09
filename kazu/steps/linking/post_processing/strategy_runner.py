import logging
from typing import List, Tuple, Iterable, Dict, FrozenSet

from kazu.data.data import (
    Document,
    Entity,
    SynonymTermWithMetrics,
)
from kazu.modelling.database.in_memory_db import MetadataDatabase
from kazu.steps.linking.post_processing.mapping_strategies.strategies import MappingStrategy
from kazu.utils.grouping import sort_then_group
from kazu.utils.string_normalizer import StringNormalizer

logger = logging.getLogger(__name__)


EntityClassStrategy = Dict[str, List[MappingStrategy]]


class NamespaceStrategyList:
    """
    simple container class to manage per ent_class strategies (EntityClassStrategy), and default strategies if no
    ent class strategy is configured. Could be managed with a Dict, but makes __init__ signature of StrategyRunner
    even more complicated than it already is?
    """

    def __init__(
        self,
        ent_class_strategies: EntityClassStrategy,
        default_strategies: List[MappingStrategy],
    ):
        """

        :param ent_class_strategies: per class strategies
        :param default_strategies: default strategies
        """
        self.default_strategies = default_strategies
        self.ent_class_strategies = ent_class_strategies

    def get_strategies_for_entity_class(self, entity_class: str) -> List[MappingStrategy]:
        return self.ent_class_strategies.get(entity_class, self.default_strategies)


class StrategyRunner:
    """
    This is a complex class, designed to co-ordinate the running of various strategies over a document, with the end
    result producing mappings (grounding) for entities. Strategies that produce mappings may depend on the changing
    state of the Document, depending on whether other strategies are successful or not, hence why their precise
    co-ordination is crucial. Specifically we want the strategies that have higher precision to run before lower
    precision ones.

    Beyound the precision of the strategy itself, the variables to consider are:

    1) the NER system (a.k.a namespace), in that different systems vary in terms of precision and recall for detecting
        entity spans
    2) whether an entity is symbolic or not (generally speaking, noun phrases should be easier to normalise than
        symbolic mentions, as there is more information to work with
    3) what SynonymTerms are associated with the entity, and from which parser they originated from

    This __call__ method of this class operates as follows:

    1) group entities by order of NER namespace
    2) sub-group these entities by whether they are symbolic or not
    3) sub-group these entities again by Entity.match and Entity.entity_class
    4) get the appropriate list of strategies to run against this sub group
    5) group the SynonymTermWithMetrics associated with this subgroup by their parser_name, filtered by any mappings
        from this parser_name already attached to the Entity
    6) run the next strategy in this list for every group of SynonymTermWithMetrics, and attach any resulting mappings
        to the relevant entity group
    7) repeat 5)-6) for every entity group from 4), until either a) mappings are produced for each parser_name
        associated with the SynonymTermWithMetrics attached to the Entity, or b) there are no more strategies to run

    """

    def __init__(
        self,
        symbolic_strategies: Dict[str, NamespaceStrategyList],
        non_symbolic_strategies: Dict[str, NamespaceStrategyList],
        ner_namespace_processing_order: List[str],
    ):
        """

        :param non_symbolic_strategies: mapping of NER namespace to a NamespaceStrategyList for non symbolic entities
        :param symbolic_strategies: mapping of NER namespace to a NamespaceStrategyList for symbolic entities
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
        """
        group entities by configured namespace order, split by symbolism, then run
            self.execute_hit_post_processing_strategies
        :param doc:
        :return:
        """
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
        This method executes parts 3 -7 in the class DocString
        :param ents_needing_mappings:
        :param document:
        :param namespace_strategy_list:
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
                    MappingStrategy
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
