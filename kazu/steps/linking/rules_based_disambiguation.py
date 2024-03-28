import logging
from collections import defaultdict
from enum import auto
from typing import Any, Literal, Optional

from spacy.matcher import Matcher
from spacy.tokens import Span


from kazu.data import Document, Entity, Section, AutoNameEnum
from kazu.steps import document_iterating_step, Step
from kazu.utils.spacy_pipeline import (
    SpacyPipelines,
    BASIC_PIPELINE_NAME,
    basic_spacy_pipeline,
)
from kazu.utils.spacy_object_mapper import KazuToSpacyObjectMapper

SpacyMatcherRules = list[list[dict[str, Any]]]
MaybeSpacyMatcherRules = Optional[SpacyMatcherRules]
logger = logging.getLogger(__name__)
TPOrFP = Literal["tp", "fp"]
TPOrFPMatcher = dict[TPOrFP, Matcher]
MatcherMentionRules = dict[str, dict[str, dict[TPOrFP, MaybeSpacyMatcherRules]]]
MatcherClassRules = dict[str, dict[TPOrFP, MaybeSpacyMatcherRules]]
MentionMatchers = dict[str, dict[str, TPOrFPMatcher]]
ClassMatchers = dict[str, TPOrFPMatcher]
_KeyToMatcherResults = dict[tuple[str, str], bool]


class MatcherResult(AutoNameEnum):
    HIT = auto()
    MISS = auto()
    NOT_CONFIGURED = auto()


class RulesBasedEntityClassDisambiguationFilterStep(Step):
    """Removes instances of :class:`.Entity` from :class:`.Section`\\s that don't meet
    rules based disambiguation requirements in at least one location in the document.

    This step utilises spaCy `Matcher <https://spacy.io/api/matcher>`_
    rules to determine whether an entity class and or/mention entities are valid or not.
    These Matcher rules operate on the sentence in which each mention under consideration
    is located.

    Rules can have both true positive and false positive aspects. If defined, that
    aspect MUST be correct at least once in the document for all entities with the same
    key (composed of the matched string and entity class) to be valid.

    Non-contiguous entities are evaluated on the full span of the text they cover, rather
    than the specific tokens.
    """

    def __init__(
        self,
        class_matcher_rules: MatcherClassRules,
        mention_matcher_rules: MatcherMentionRules,
    ):
        """

        :param class_matcher_rules: these should follow the format:

            .. code-block:: python

                {
                    "<entity class>": {
                        "<tp or fp (for true positive or false positive rules respectively>": [
                            "<a list of rules>",
                            "<according to the spaCy pattern matcher syntax>",
                        ]
                    }
                }
        :param mention_matcher_rules: these should follow the format:

            .. code-block:: python

                {
                    "<entity class>": {
                        "<mention to disambiguate>": {
                            "<tp or fp>": [
                                "<a list of rules>",
                                "<according to the spaCy pattern matcher syntax>",
                            ]
                        }
                    }
                }
        """

        self.class_matcher_rules = class_matcher_rules
        self.mention_matcher_rules = mention_matcher_rules

        self.entity_classes_used_in_rules = self._calculate_custom_extensions_used_in_rules(
            class_matcher_rules, mention_matcher_rules
        )

        self.mapper = KazuToSpacyObjectMapper(self.entity_classes_used_in_rules)

        self.spacy_pipelines = SpacyPipelines()
        self.spacy_pipelines.add_from_func(BASIC_PIPELINE_NAME, basic_spacy_pipeline)
        self.spacy_pipelines.add_reload_callback_func(
            BASIC_PIPELINE_NAME, self._build_class_matchers
        )
        self.spacy_pipelines.add_reload_callback_func(
            BASIC_PIPELINE_NAME, self._build_mention_matchers
        )
        self._build_class_matchers()
        self._build_mention_matchers()

    @staticmethod
    def _calculate_custom_extensions_used_in_rules(
        class_matcher_rules: MatcherClassRules,
        mention_matcher_rules: MatcherMentionRules,
    ) -> set[str]:
        """Calculate the spaCy 'custom extensions' used in rules.

        This considers both the class matcher rules and mention matcher rules passed to
        the class constructor.

        These are all expected to be entity classes, since these are the only attributes
        that will be populated by KazuToSpacyObjectMapper, so using any other custom
        extensions would have no effect.
        """
        spacy_rules: SpacyMatcherRules = []
        custom_extensions: set[str] = set()
        for tp_or_fp_rule_struct in class_matcher_rules.values():
            for rules in tp_or_fp_rule_struct.values():
                if rules is not None:
                    spacy_rules.extend(rules)

        for mention_rule_struct in mention_matcher_rules.values():
            for tp_or_fp_rule_struct in mention_rule_struct.values():
                for rules in tp_or_fp_rule_struct.values():
                    if rules is not None:
                        spacy_rules.extend(rules)

        for rule in spacy_rules:
            for token_dict in rule:
                if (current_custom_extensions := token_dict.get("_")) is not None:
                    custom_extensions.update(current_custom_extensions.keys())

        return custom_extensions

    def _build_class_matchers(self) -> None:
        result: ClassMatchers = {}
        for class_name, rules in self.class_matcher_rules.items():
            for rule_type, rule_instances in rules.items():
                if rule_instances is not None:
                    matcher = Matcher(self.spacy_pipelines.get_model(BASIC_PIPELINE_NAME).vocab)
                    matcher.add(f"{class_name}_{rule_type}", rule_instances)
                    result.setdefault(class_name, {})[rule_type] = matcher
        self.class_matchers = result

    def _build_mention_matchers(self) -> None:
        result: MentionMatchers = {}
        rule_type: TPOrFP
        for class_name, target_syn_dict in self.mention_matcher_rules.items():
            for target_syn, rules in target_syn_dict.items():
                for rule_type, rule_instances in rules.items():
                    if rule_instances is not None:
                        matcher = Matcher(self.spacy_pipelines.get_model(BASIC_PIPELINE_NAME).vocab)
                        matcher.add(f"{class_name}_{target_syn}_{rule_type}", rule_instances)
                        result_for_class = result.setdefault(class_name, {})
                        result_for_class_and_target_syn = result_for_class.setdefault(
                            target_syn, {}
                        )
                        result_for_class_and_target_syn[rule_type] = matcher
        self.mention_matchers = result

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        ent_tp_class_results: _KeyToMatcherResults = {}
        ent_fp_class_results: _KeyToMatcherResults = {}
        ent_tp_mention_results: _KeyToMatcherResults = {}
        ent_fp_mention_results: _KeyToMatcherResults = {}

        class_tp_is_configured_for_key = {}
        class_fp_is_configured_for_key = {}
        mention_tp_is_configured_for_key = {}
        mention_fp_is_configured_for_key = {}

        # keep track of only entities that could be affected by this step, so we don't need
        # to loop over everything later
        section_to_ents_under_consideration: defaultdict[Section, set[Entity]] = defaultdict(set)

        for section in doc.sections:
            ent_to_span = self.mapper(section)

            for entity in section.entities:
                if entity not in ent_to_span:
                    # there's a chance that an entity can't be matched to a spacy
                    # span, e.g. if the entity only covers non-token text.
                    # We skip the entity in this case.
                    continue
                entity_class = entity.entity_class

                entity_match = entity.match
                key = (
                    entity_match,
                    entity_class,
                )
                maybe_class_matchers = self.class_matchers.get(entity_class)
                maybe_mention_matchers = self.mention_matchers.get(entity_class, {}).get(
                    entity.match
                )
                # if neither class nor mention matcher defined (the usual case), just continue
                if maybe_class_matchers is None and maybe_mention_matchers is None:
                    continue

                section_to_ents_under_consideration[section].add(entity)
                tp_class_result, fp_class_result = self._check_tp_fp_matcher_rules(
                    entity, ent_to_span, maybe_class_matchers
                )
                if tp_class_result is MatcherResult.NOT_CONFIGURED:
                    class_tp_is_configured_for_key[key] = False
                else:
                    class_tp_is_configured_for_key[key] = True
                if fp_class_result is MatcherResult.NOT_CONFIGURED:
                    class_fp_is_configured_for_key[key] = False
                else:
                    class_fp_is_configured_for_key[key] = True

                ent_tp_class_results[key] = (
                    ent_tp_class_results.get(key, False) or tp_class_result is MatcherResult.HIT
                )
                ent_fp_class_results[key] = (
                    ent_fp_class_results.get(key, False) or fp_class_result is MatcherResult.HIT
                )

                tp_mention_result, fp_mention_result = self._check_tp_fp_matcher_rules(
                    entity, ent_to_span, maybe_mention_matchers
                )
                if tp_mention_result is MatcherResult.NOT_CONFIGURED:
                    mention_tp_is_configured_for_key[key] = False
                else:
                    mention_tp_is_configured_for_key[key] = True
                if fp_mention_result is MatcherResult.NOT_CONFIGURED:
                    mention_fp_is_configured_for_key[key] = False
                else:
                    mention_fp_is_configured_for_key[key] = True

                ent_tp_mention_results[key] = (
                    ent_tp_mention_results.get(key, False) or tp_mention_result is MatcherResult.HIT
                )
                ent_fp_mention_results[key] = (
                    ent_fp_mention_results.get(key, False) or fp_class_result is MatcherResult.HIT
                )

        for section, ents in section_to_ents_under_consideration.items():
            for ent in ents:
                key = (
                    ent.match,
                    ent.entity_class,
                )
                if (
                    (class_fp_is_configured_for_key[key] and ent_fp_class_results[key])
                    or (class_tp_is_configured_for_key[key] and not ent_tp_class_results[key])
                    or (mention_fp_is_configured_for_key[key] and ent_fp_mention_results[key])
                    or (mention_tp_is_configured_for_key[key] and not ent_tp_mention_results[key])
                ):
                    section.entities.remove(ent)

    @staticmethod
    def _check_matcher(context: Span, maybe_matcher: Optional[Matcher]) -> MatcherResult:
        if maybe_matcher is None:
            return MatcherResult.NOT_CONFIGURED
        if maybe_matcher(context):
            return MatcherResult.HIT
        else:
            return MatcherResult.MISS

    @staticmethod
    def _check_tp_fp_matcher_rules(
        entity: Entity, ent_to_span: dict[Entity, Span], matchers: Optional[TPOrFPMatcher]
    ) -> tuple[MatcherResult, MatcherResult]:
        if matchers is None:
            return MatcherResult.NOT_CONFIGURED, MatcherResult.NOT_CONFIGURED
        span = ent_to_span[entity]
        context = span.sent

        tp_matcher = matchers.get("tp")
        tp_result = RulesBasedEntityClassDisambiguationFilterStep._check_matcher(
            context, tp_matcher
        )
        fp_matcher = matchers.get("fp")
        fp_result = RulesBasedEntityClassDisambiguationFilterStep._check_matcher(
            context, fp_matcher
        )
        return tp_result, fp_result
