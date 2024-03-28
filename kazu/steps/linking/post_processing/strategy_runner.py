import logging
from collections.abc import Iterable
from copy import deepcopy
from itertools import groupby
from typing import Optional

from kazu.data import (
    Document,
    Entity,
    LinkingCandidate,
    Mapping,
    MentionConfidence,
)
from kazu.steps.linking.post_processing.mapping_strategies.strategies import MappingStrategy
from kazu.steps.linking.post_processing.xref_manager import CrossReferenceManager
from kazu.utils.grouping import sort_then_group
from kazu.utils.string_normalizer import StringNormalizer

logger = logging.getLogger(__name__)


EntityClassStrategy = dict[str, list[MappingStrategy]]


# hashable representation of all identical (for the purposes of mapping) entities in the document
EntityKey = tuple[str, str, str, frozenset[LinkingCandidate]]


def entity_to_entity_key(
    e: Entity,
) -> EntityKey:
    return (
        e.match,
        e.match_norm,
        e.entity_class,
        frozenset(e.linking_candidates),
    )


class ConfidenceLevelStrategyExecution:
    """The role of this class is to track which entities have had mappings successfully
    resolved, and which require the application of further strategies.

    This is handled via tracking a dictionary of EntityKey to sets of parser names.

    See further details in the __call__ docstring.
    """

    def __init__(
        self,
        ent_class_strategies: EntityClassStrategy,
        default_strategies: list[MappingStrategy],
        stop_on_success: bool = False,
    ):
        """

        :param ent_class_strategies: per class strategies
        :param default_strategies: default strategies
        :param stop_on_success: If ``True``, stop after the first
            successful strategy, even if some parsers remain
            unresolved. Otherwise, keep running until all parsers
            are resolved (or all relevant strategies have been tried).
        """
        self.stop_on_success = stop_on_success
        self.default_strategies = default_strategies
        self.ent_class_strategies = ent_class_strategies
        self.unresolved_parsers: dict[EntityKey, set[str]] = {}
        self.entity_mapped: dict[EntityKey, bool] = {}

    @property
    def longest_mapping_strategy_list_size(self) -> int:
        return max(
            (
                len(self.default_strategies),
                *(len(strategies) for strategies in self.ent_class_strategies.values()),
            )
        )

    def get_strategies_for_entity_class(self, entity_class: str) -> list[MappingStrategy]:
        return self.ent_class_strategies.get(entity_class, self.default_strategies)

    def _get_unresolved_parsers(self, entity_key: EntityKey, entity: Entity) -> set[str]:

        maybe_unresolved_parsers = self.unresolved_parsers.get(entity_key, None)
        if maybe_unresolved_parsers is not None:
            return maybe_unresolved_parsers
        else:
            unresolved_parsers = set(x.parser_name for x in entity.linking_candidates)
            self.unresolved_parsers[entity_key] = unresolved_parsers
            return unresolved_parsers

    def __call__(
        self, entity: Entity, strategy_index: int, document: Document
    ) -> Iterable[Mapping]:
        """Conditionally execute a mapping strategy over an entity.

        :param entity: entity to process
        :param strategy_index: index of strategy to run that is configured for this
            entity class
        :param document: originating Document
        :return:
        """
        strategy_list: list[MappingStrategy] = self.get_strategies_for_entity_class(
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
            elif self.stop_on_success and self.entity_mapped.get(entity_key, False):
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
                candidates_to_consider = (
                    t for t in entity.linking_candidates if t.parser_name in unresolved_parsers
                )
                candidates_by_parser = sort_then_group(
                    candidates_to_consider, key_func=lambda x: x.parser_name
                )

                for parser_name, candidates_this_parser in candidates_by_parser:
                    for mapping in strategy(
                        ent_match=entity.match,
                        ent_match_norm=entity.match_norm,
                        candidates={
                            candidate: entity.linking_candidates[candidate]
                            for candidate in candidates_this_parser
                        },  # pass a shallow copy so that delegated implementations don't have to worry about data corruption
                        document=document,
                    ):
                        self.unresolved_parsers[entity_key].discard(mapping.parser_name)
                        self.entity_mapped[entity_key] = True
                        yield mapping

    def reset(self):
        """Clear state, ready for another execution.

        Should be called when the underlying :class:`.Document` has
        changed.
        """
        self.unresolved_parsers.clear()
        self.entity_mapped.clear()


class StrategyRunner:
    """This is a complex class, designed to co-ordinate the running of various
    strategies over a document, with the end result producing mappings (grounding) for
    entities. Strategies that produce mappings may depend on the changing state of the
    Document, depending on whether other strategies are successful or not, hence why
    their precise co-ordination is crucial. Specifically we want the strategies that
    have higher precision to run before lower precision ones.

    Beyond the precision of the strategy itself, the variables to consider are:

    1. the confidence of the NER systems in the match, in that different systems vary in terms of precision and recall for detecting
       entity spans.
    2. what LinkingCandidates are associated with the entity, and from which parser they originated from.

    The __call__ method of this class operates as follows:

    1. group entities by order of :class:`.MentionConfidence`\\.
    2. sub-group these entities again by :attr:`.Entity.match` and
       :attr:`.Entity.entity_class`\\ .
    3. divide these entities by whether they are symbolic or not.
    4. identify the maximum number of strategies that 'could' run.
    5. get the appropriate :class:`ConfidenceLevelStrategyExecution` to run against this sub group.
    6. group the entities from 5. by EntityKey (i.e. a hashable representation of unique information required for
       mapping.
    7. conditionally execute the next strategy out of the maximum possible (from 4), and attach any resulting mappings
       to the relevant entity group. Note, the :class:`ConfidenceLevelStrategyExecution` is responsible for deciding whether
       a strategy is executed or not.
    """

    def __init__(
        self,
        symbolic_strategies: dict[str, ConfidenceLevelStrategyExecution],
        non_symbolic_strategies: dict[str, ConfidenceLevelStrategyExecution],
        cross_ref_managers: Optional[list[CrossReferenceManager]] = None,
    ):
        """


        :param symbolic_strategies: mapping of mention confidence to a :class:`ConfidenceLevelStrategyExecution` for symbolic
            entities
        :param non_symbolic_strategies: mapping of mention confidence to a :class:`ConfidenceLevelStrategyExecution` for
            non-symbolic entities
        :param cross_ref_managers: list of managers that will be applied to any created mappings, attempting to create
            xreferences
        """
        self.symbolic_strategies = {MentionConfidence[k]: v for k, v in symbolic_strategies.items()}
        self.non_symbolic_strategies = {
            MentionConfidence[k]: v for k, v in non_symbolic_strategies.items()
        }
        self.cross_ref_managers = cross_ref_managers

    @staticmethod
    def group_entities_by_symbolism(
        entities: Iterable[Entity],
    ) -> tuple[list[Entity], list[Entity]]:
        """Groups entities into symbolic and non-symbolic lists, so they can be
        processed separately.

        :param entities:
        :return:
        """
        symbolic: list[Entity] = []
        non_symbolic: list[Entity] = []
        grouped_by_match = sort_then_group(
            entities,
            key_func=lambda x: (
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

    def __call__(self, doc: Document) -> None:
        """Run relevant strategies to decide what mappings to create.

        Generally speaking, noun phrases should be easier to normalise than symbolic mentions, as there is more
        information to work with. Therefore, we group entities by mention confidence, split by symbolism, then
        run :meth:`execute_hit_post_processing_strategies`\\ .

        :param doc:
        :return:
        """

        # do a separate sorted and groupby call (rather than our sort_then_group utility)
        # so we can do all the sorting we need in one go
        # we inverse the sign of mention_confidence so that we process high confidence
        # hits first
        sorted_entities = sorted(
            doc.get_entities(),
            key=lambda ent: (
                -ent.mention_confidence,
                *entity_to_entity_key(ent),
            ),
        )
        entities_grouped_by_confidence = groupby(
            sorted_entities, key=lambda ent: ent.mention_confidence
        )

        for mention_confidence, entities in entities_grouped_by_confidence:
            logger.debug("mapping entities for confidence %s", mention_confidence)
            symbolic_entities, non_symbolic_entities = self.group_entities_by_symbolism(
                entities=entities
            )

            maybe_non_symbolic_strategies = self.non_symbolic_strategies.get(mention_confidence)
            if maybe_non_symbolic_strategies is not None:
                self.execute_hit_post_processing_strategies(
                    non_symbolic_entities, doc, maybe_non_symbolic_strategies
                )
            else:
                logger.warning(
                    "No %s configured for %s ",
                    ConfidenceLevelStrategyExecution.__name__,
                    mention_confidence,
                )
            maybe_symbolic_strategies = self.symbolic_strategies.get(mention_confidence)
            if maybe_symbolic_strategies is not None:
                self.execute_hit_post_processing_strategies(
                    symbolic_entities, doc, maybe_symbolic_strategies
                )
            else:
                logger.warning(
                    "No %s configured for %s ",
                    ConfidenceLevelStrategyExecution.__name__,
                    mention_confidence,
                )

    def execute_hit_post_processing_strategies(
        self,
        ents_needing_mappings: list[Entity],
        document: Document,
        confidence_strategy_execution: ConfidenceLevelStrategyExecution,
    ) -> None:
        """
        This method executes parts 5 - 7 in the class Docstring.

        :param ents_needing_mappings: Expects entities to already be sorted based on :func:`entity_to_entity_key`\\ .
        :param document:
        :param confidence_strategy_execution:
        :return:
        """
        try:
            strategy_max_index = confidence_strategy_execution.longest_mapping_strategy_list_size

            groups = [
                list(ents)
                for _entity_key, ents in groupby(ents_needing_mappings, key=entity_to_entity_key)
            ]

            for i in range(0, strategy_max_index):
                for entity_group in groups:
                    reference_entity = next(iter(entity_group))
                    for mapping in confidence_strategy_execution(
                        entity=reference_entity,
                        strategy_index=i,
                        document=document,
                    ):
                        xref_mappings: set[Mapping] = set()
                        if self.cross_ref_managers is not None:
                            for xref_manager in self.cross_ref_managers:
                                xref_mappings.update(
                                    xref_manager.create_xref_mappings(mapping=mapping)
                                )

                        for entity in entity_group:
                            entity.mappings.add(deepcopy(mapping))
                            entity.mappings.update(deepcopy(xref_mappings))
                        logger.debug(
                            "mapping created: original string: %s, mapping: %s, cross-references: %s",
                            reference_entity.match,
                            mapping,
                            xref_mappings,
                        )
        finally:
            # in case exception is thrown - always reset state
            confidence_strategy_execution.reset()
