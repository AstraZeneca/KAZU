import copy
import logging
from itertools import groupby
from typing import List, Tuple, Iterable, Dict, FrozenSet, Set

from kazu.data.data import (
    Document,
    Entity,
    SynonymTermWithMetrics,
    Mapping,
)
from kazu.modelling.database.in_memory_db import MetadataDatabase
from kazu.steps.linking.post_processing.mapping_strategies.strategies import MappingStrategy
from kazu.utils.grouping import sort_then_group
from kazu.utils.string_normalizer import StringNormalizer

logger = logging.getLogger(__name__)


EntityClassStrategy = Dict[str, List[MappingStrategy]]


# hashable representation of all identical (for the purposes of mapping) entities in the document
EntityKey = Tuple[str, str, str, FrozenSet[SynonymTermWithMetrics]]


def entity_to_entity_key(
    e: Entity,
) -> EntityKey:
    return (
        e.match,
        e.match_norm,
        e.entity_class,
        frozenset(e.syn_term_to_synonym_terms),
    )


class NamespaceStrategyExecution:
    """
    The role of a NamespaceStrategyExecution is to track which entities have had mappings successfully resolved,
    and which require the application of further strategies. This is handled via tracking a dictionary of
    EntityKey:Set[<parser name>]. See further details in the __call__ docstring
    """

    def __init__(
        self,
        ent_class_strategies: EntityClassStrategy,
        default_strategies: List[MappingStrategy],
        stop_on_success: bool = False,
    ):
        """

        :param ent_class_strategies: per class strategies
        :param default_strategies: default strategies
        """
        self.stop_on_success = stop_on_success
        self.default_strategies = default_strategies
        self.ent_class_strategies = ent_class_strategies
        self.unresolved_parsers: Dict[EntityKey, Set[str]] = {}

    @property
    def longest_mapping_strategy_list_size(self):
        return max(
            len(self.default_strategies),
            *(len(strategies) for strategies in self.ent_class_strategies.values()),
        )

    def get_strategies_for_entity_class(self, entity_class: str) -> List[MappingStrategy]:
        return self.ent_class_strategies.get(entity_class, self.default_strategies)

    def _get_unresolved_parsers(self, entity_key: EntityKey, entity: Entity) -> Set[str]:

        maybe_unresolved_parsers = self.unresolved_parsers.get(entity_key, None)
        if maybe_unresolved_parsers is not None:
            return maybe_unresolved_parsers
        else:
            unresolved_parsers = set(x.parser_name for x in entity.syn_term_to_synonym_terms)
            self.unresolved_parsers[entity_key] = unresolved_parsers
            return unresolved_parsers

    def __call__(
        self, entity: Entity, strategy_index: int, document: Document
    ) -> Iterable[Mapping]:
        """
        conditionally execute a mapping strategy over an entity

        :param entity: entity to process
        :param strategy_index: index of strategy to run that is configured for this entity class
        :param document: originating Document
        :return:
        """
        strategy_list: List[MappingStrategy] = self.get_strategies_for_entity_class(
            entity_class=entity.entity_class
        )
        if strategy_index > len(strategy_list) - 1:
            logger.debug("no more strategies this class")
        else:
            strategy = strategy_list[strategy_index]
            entity_key = entity_to_entity_key(entity)
            # we keep track of which entities have resolved mappings to specific parsers, so we don't run lower
            # ranked strategies if we don't need to
            unresolved_parsers = self._get_unresolved_parsers(entity_key, entity)
            if len(unresolved_parsers) == 0:
                logger.debug(
                    f"will not run strategy {strategy.__class__.__name__} on class :<{entity.entity_class}>, match: "
                    f"<{entity.match}> as all parsers have been resolved"
                )
            elif self.stop_on_success:
                logger.debug(
                    f"will not run strategy {strategy.__class__.__name__} on class :<{entity.entity_class}>, match: "
                    f"<{entity.match}> as entity has been resolved to another parser and stop_on_success: "
                    f"{self.stop_on_success}"
                )
            else:
                logger.debug(
                    f"running strategy {strategy.__class__.__name__} on class :<{entity.entity_class}>, match: "
                    f"<{entity.match}> "
                )
                strategy.prepare(document)
                terms_to_consider = (
                    t
                    for t in entity.syn_term_to_synonym_terms
                    if t.parser_name in unresolved_parsers
                )
                terms_by_parser = sort_then_group(
                    terms_to_consider, key_func=lambda x: x.parser_name
                )

                for parser_name, terms_this_parser in terms_by_parser:
                    terms_this_parser_set = frozenset(terms_this_parser)
                    for mapping in strategy(
                        ent_match=entity.match,
                        ent_match_norm=entity.match_norm,
                        terms=terms_this_parser_set,
                        document=document,
                    ):
                        self.unresolved_parsers[entity_key].discard(mapping.parser_name)
                        yield mapping

    def reset(self):
        """
        clear state, ready for another execution. Should be called when the underlying :class:`.Document` has changed

        :return:
        """
        self.unresolved_parsers.clear()


class StrategyRunner:
    """
    This is a complex class, designed to co-ordinate the running of various strategies over a document, with the end
    result producing mappings (grounding) for entities. Strategies that produce mappings may depend on the changing
    state of the Document, depending on whether other strategies are successful or not, hence why their precise
    co-ordination is crucial. Specifically we want the strategies that have higher precision to run before lower
    precision ones.

    Beyond the precision of the strategy itself, the variables to consider are:

    1) the NER system (a.k.a namespace), in that different systems vary in terms of precision and recall for detecting
       entity spans.
    2) what SynonymTerms are associated with the entity, and from which parser they originated from.

    The __call__ method of this class operates as follows:

    1) group entities by order of NER namespace.
    2) sub-group these entities again by :attr:`.Entity.match` and :attr:`.Entity.entity_class`.
    3) divide these entities by whether they are symbolic or not.
    4) identify the maximum number of strategies that 'could' run.
    5) get the appropriate :class:`.NamespaceStrategyExecution` to run against this sub group
    6) group the entities from 5) by EntityKey (i.e. a hashable representation of unique information .required for
       mapping.
    7) conditionally execute the next strategy out of the maximum possible (from 4), and attach any resulting mappings
       to the relevant entity group. Note, the :class:`NamespaceStrategyExecution` is responsible for deciding whether
       a strategy is executed or not.
    """

    def __init__(
        self,
        symbolic_strategies: Dict[str, NamespaceStrategyExecution],
        non_symbolic_strategies: Dict[str, NamespaceStrategyExecution],
        ner_namespace_processing_order: List[str],
    ):
        """

        :param non_symbolic_strategies: mapping of NER namespace to a :class:`NamespaceStrategyExecution` for
            non-symbolic entities
        :param symbolic_strategies: mapping of NER namespace to a :class:`NamespaceStrategyExecution` for symbolic
            entities
        :param ner_namespace_processing_order: Entities will be mapped in this namespace order. This is
            useful if you have a high precision, low recall NER namespace, combined with a low precision high recall
            namespace, as the mapping info derived from the high precision NER namespace can be used with a high
            precision strategy for the low precision NER namespace
        """
        self.non_symbolic_strategies = non_symbolic_strategies
        self.symbolic_strategies = symbolic_strategies
        self.ner_namespace_processing_order = ner_namespace_processing_order
        self.metadata_db = MetadataDatabase()

        if self.ner_namespace_processing_order is not None:
            self.ner_namespace_to_index = {
                ns: ind for ind, ns in enumerate(self.ner_namespace_processing_order)
            }
            self.get_namespace_sort_key = lambda ns: self.ner_namespace_to_index[ns]
        else:
            self.get_namespace_sort_key = lambda ns: ns

    @staticmethod
    def group_entities_by_symbolism(
        entities: Iterable[Entity],
    ) -> Tuple[List[Entity], List[Entity]]:
        """
        groups entities into symbolic and non-symbolic forms, so they can be processed separately.
        Expects an already sorted list of entities, since we only call this after a sort is required
        elsewhere. However, it will still work with an unsorted list, it will just call
        :meth:`.StringNormalizer.classify_symbolic` more times than necessary.

        :param entities:
        :return:
        """
        symbolic: List[Entity] = []
        non_symbolic: List[Entity] = []
        grouped_by_match = groupby(
            entities,
            key=lambda x: (
                x.match,
                x.entity_class,
            ),
        )
        for (match_str, entity_class), ent_iter in grouped_by_match:
            if StringNormalizer.classify_symbolic(match_str, entity_class=entity_class):
                symbolic.extend(ent_iter)
            else:
                non_symbolic.extend(ent_iter)
        return symbolic, non_symbolic

    def __call__(self, doc: Document):
        """
        generally speaking, noun phrases should be easier to normalise than symbolic mentions, as there is more
        information to work with. Therefore, we group entities by configured namespace order, split by symbolism, then
        run :meth:`execute_hit_post_processing_strategies`

        :param doc:
        :return:
        """
        # do a separate sorted and groupby call (rather than our sort_then_group utility)
        # so we can do all the sorting we need in one go
        sorted_entities = sorted(
            doc.get_entities(),
            key=lambda ent: (
                self.get_namespace_sort_key(ent.namespace),
                *entity_to_entity_key(ent),
            ),
        )
        # add in ent.namespace, so we have it available in the group key.
        # It won't affect the sorting since the first element of the tuple will be the same
        # for all ents with the same namespace
        entities_grouped_by_namespace_order = groupby(
            sorted_entities,
            key=lambda ent: (self.get_namespace_sort_key(ent.namespace), ent.namespace),
        )

        for (_namespace_sort_key, namespace), entities in entities_grouped_by_namespace_order:
            logger.debug("mapping entities for namespace %s", namespace)
            symbolic_entities, non_symbolic_entities = self.group_entities_by_symbolism(entities)
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
        namespace_strategy_execution: NamespaceStrategyExecution,
    ):
        """
        This method executes parts 5 - 7 in the class DocString

        :param ents_needing_mappings: Expects entities to already be sorted based on :func:`entity_to_entity_key`
        :param document:
        :param namespace_strategy_execution:
        :return:
        """
        namespace_strategy_execution.reset()
        strategy_max_index = namespace_strategy_execution.longest_mapping_strategy_list_size

        groups = [
            list(ents)
            for _entity_key, ents in groupby(ents_needing_mappings, key=entity_to_entity_key)
        ]

        for i in range(0, strategy_max_index):
            for entity_group in groups:
                reference_entity = next(iter(entity_group))
                for mapping in namespace_strategy_execution(
                    entity=reference_entity,
                    strategy_index=i,
                    document=document,
                ):
                    for entity in entity_group:
                        entity.mappings.add(copy.deepcopy(mapping))
                    logger.debug(
                        "mapping created: original string: %s, mapping: %s",
                        reference_entity.match,
                        mapping,
                    )
