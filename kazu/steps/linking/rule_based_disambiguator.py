import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import Optional, Any

import spacy
from kazu.data.data import Document, Entity, Section
from kazu.utils.grouping import sort_then_group
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from kazu.steps import Step, document_iterating_step
from kazu.ontology_matching.assemble_pipeline import (  # noqa: F401 # we need this import to register the spacy component
    KazuCustomEnglish,
)


logger = logging.getLogger(__name__)
TPOrFP = Literal["tp", "fp"]
CooccurrenceListData = dict[str, dict[str, dict[TPOrFP, Optional[list[str]]]]]
CooccurrenceSetData = dict[str, dict[str, dict[str, set[str]]]]
ClassMatchers = dict[str, dict[str, Matcher]]
MatcherClassRules = dict[str, dict[TPOrFP, list[list[dict[str, Any]]]]]


class RulesBasedEntityClassDisambiguationFilterStep(Step):
    """Removes instances of :class:`.Entity` from
    :class:`.Section` that don't meet rules based
    disambiguation requirements in at least one location in the document.

    This step can utilise both Spacy `Matcher <https://spacy.io/api/matcher>`_
    rules and sentence based keyword co-occurrence information to determine whether
    entities are valid or not.

    Rules can have both true positive and false positive aspects.

    `Matcher <https://spacy.io/api/matcher>`_ rules operate on the entity class level
    (i.e. apply to whole classes of an entity), whereas co-occurrence rules operate
    on the mention level (i.e. does a specific mention of an entity co-occur with some expected words,
    in the same sentence).

    """

    def __init__(
        self, class_matcher_rules: MatcherClassRules, cooccurrence_rules: CooccurrenceListData
    ):
        """

        :param class_matcher_rules: these should follow the format:
            .. code-block::

                {<entity class>:
                    {<tp or fp (for true positive or false positive rules respectively>:
                        [a list of rules according to the spacy pattern matcher syntax]
                    }
                }
        :param cooccurrence_rules: these should follow the format:
            .. code-block::

                {<entity class>:
                    {<mention to disambiguate>:
                        {<tp or fp (for true positive or false positive rules respectively>:
                            [<a list of keywords to disambiguate the context>]
                        }
                    }
                }
        """
        self.nlp = spacy.blank("kazu_custom_en")
        self.nlp.add_pipe("sentencizer")
        self.class_matchers: ClassMatchers = self._build_class_matchers(class_matcher_rules)
        self.cooccurrence_rules: CooccurrenceSetData = self._build_cooccurrence_rules(
            cooccurrence_rules
        )

        for entity_class in self.class_matchers:
            Token.set_extension(entity_class, default=False, force=True)

        for entity_class in self.cooccurrence_rules:
            Token.set_extension(entity_class, default=False, force=True)

    @staticmethod
    def _build_cooccurrence_rules(coocurance_data: CooccurrenceListData) -> CooccurrenceSetData:
        result: dict = coocurance_data.copy()
        for class_name, term_data in coocurance_data.items():
            for term, rule_type_and_terms in term_data.items():
                for rule_type, terms_list in rule_type_and_terms.items():
                    if terms_list is not None:
                        result[class_name][term][rule_type] = set(terms_list)
        return result

    def _build_class_matchers(self, class_rules: MatcherClassRules) -> ClassMatchers:
        result: defaultdict[str, dict[str, Matcher]] = defaultdict(dict)
        for class_name, rules in class_rules.items():
            for rule_type, rule_instances in rules.items():
                if rule_instances is not None:
                    matcher = Matcher(self.nlp.vocab)
                    matcher.add(f"{class_name}_{rule_type}", rule_instances)
                    result[class_name][rule_type] = matcher
        return dict(result)

    def _cooccurrence_found(self, context: str, cooc_strings: set) -> bool:
        return any(x in context for x in cooc_strings)

    def _run_class_matchers(
        self,
        entity_class: str,
        doc: Doc,
    ) -> Iterable[bool]:
        maybe_class_matchers = self.class_matchers.get(entity_class)
        if maybe_class_matchers is None:
            yield True
        else:
            tp_matcher = maybe_class_matchers.get("tp")
            fp_matcher = maybe_class_matchers.get("fp")
            if tp_matcher is None:
                yield True
            else:
                if tp_matcher(doc):
                    yield True
                else:
                    yield False

            if fp_matcher is None:
                yield True
            else:
                if fp_matcher(doc):
                    yield False
                else:
                    yield True

    def _run_cooccurrence_rules(
        self,
        entity_class: str,
        entity_match: str,
        span: Span,
    ) -> Iterable[bool]:
        maybe_cooccurrence_rules = self.cooccurrence_rules.get(entity_class, {}).get(entity_match)
        if maybe_cooccurrence_rules is None:
            yield True
        else:
            context = span.sent.text
            tp_rules = maybe_cooccurrence_rules.get("tp")
            fp_rules = maybe_cooccurrence_rules.get("fp")
            if tp_rules is None:
                yield True
            else:
                if self._cooccurrence_found(context, tp_rules):
                    yield True
                else:
                    yield False

            if fp_rules is None:
                yield True
            else:
                if self._cooccurrence_found(context, fp_rules):
                    return False
                else:
                    yield True

    @document_iterating_step
    def __call__(self, doc: Document) -> None:

        entity_mentions_needing_disambiguation: set[Entity] = set()
        entity_classes_needing_disambiguation: defaultdict[
            Section, set[tuple[Doc, str]]
        ] = defaultdict(set)
        ent_to_span: dict[Entity, Span] = {}

        for section in doc.sections:
            spacy_doc: Doc = self.nlp(section.text)
            for entity in section.entities:
                entity_class = entity.entity_class
                match = entity.match
                span = spacy_doc.char_span(
                    start_idx=entity.start, end_idx=entity.end, label=entity.entity_class
                )
                ent_to_span[entity] = span
                for token in span:
                    token._.set(entity.entity_class, True)
                if entity_class in self.class_matchers or match in self.cooccurrence_rules.get(
                    entity_class, {}
                ):
                    entity_mentions_needing_disambiguation.add(entity)
                    entity_classes_needing_disambiguation[section].add(
                        (
                            spacy_doc,
                            entity_class,
                        )
                    )

        class_keys_to_keep, maybe_class_keys_to_drop = self._check_class_matcher_rules(
            entity_classes_needing_disambiguation
        )

        maybe_mention_keys_to_drop, mention_keys_to_keep = self._check_cooccurrence_rules(
            ent_to_span, entity_mentions_needing_disambiguation
        )

        maybe_class_keys_to_drop.difference_update(class_keys_to_keep)
        maybe_mention_keys_to_drop.difference_update(mention_keys_to_keep)
        for section in doc.sections:
            for ent in list(section.entities):
                if ent.entity_class in maybe_class_keys_to_drop:
                    section.entities.remove(ent)
                elif (ent.match, ent.entity_class) in maybe_mention_keys_to_drop:
                    section.entities.remove(ent)

    def _check_cooccurrence_rules(
        self, ent_to_span: dict[Entity, Span], entity_mentions_needing_disambiguation: set[Entity]
    ) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
        maybe_mention_keys_to_drop = set()
        mention_keys_to_keep = set()
        for key, ents_iter in sort_then_group(
            entity_mentions_needing_disambiguation, key_func=lambda x: (x.match, x.entity_class)
        ):
            match, entity_class = key

            for entity in ents_iter:
                cooccurrence_results: set[bool] = set()
                cooccurrence_results.update(
                    self._run_cooccurrence_rules(
                        entity_class=entity_class,
                        entity_match=match,
                        span=ent_to_span[entity],
                    )
                )
                key = entity.match, entity.entity_class
                if False not in cooccurrence_results and True in cooccurrence_results:
                    mention_keys_to_keep.add(key)
                else:
                    maybe_mention_keys_to_drop.add(key)
        return maybe_mention_keys_to_drop, mention_keys_to_keep

    def _check_class_matcher_rules(
        self, entity_classes_needing_disambiguation: defaultdict[Section, set[tuple[Doc, str]]]
    ) -> tuple[set[str], set[str]]:
        maybe_class_keys_to_drop = set()
        class_keys_to_keep = set()
        for section, spacy_doc_and_class_set in entity_classes_needing_disambiguation.items():
            for spacy_doc, entity_class in spacy_doc_and_class_set:
                class_match_results = set(self._run_class_matchers(entity_class, spacy_doc))
                if False not in class_match_results and True in class_match_results:
                    class_keys_to_keep.add(entity_class)
                else:
                    maybe_class_keys_to_drop.add(entity_class)
        return class_keys_to_keep, maybe_class_keys_to_drop
