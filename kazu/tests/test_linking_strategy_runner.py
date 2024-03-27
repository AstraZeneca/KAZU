from typing import Optional

import pytest

from kazu.data import (
    Document,
    StringMatchConfidence,
    Entity,
    EquivalentIdSet,
    DisambiguationConfidence,
    MentionConfidence,
    CandidatesToMetrics,
    LinkingCandidate,
    LinkingMetrics,
)
from kazu.database.in_memory_db import SynonymDatabase
from kazu.ontology_preprocessing.base import IDX, SYN
from kazu.steps.linking.post_processing.disambiguation.strategies import DisambiguationStrategy
from kazu.steps.linking.post_processing.strategy_runner import (
    StrategyRunner,
    ConfidenceLevelStrategyExecution,
)
from kazu.steps.linking.post_processing.mapping_strategies.strategies import MappingStrategy
from kazu.tests.utils import DummyParser


class NoopMappingStrategy(MappingStrategy):
    def filter_candidates(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        candidates: CandidatesToMetrics,
        parser_name: str,
    ) -> CandidatesToMetrics:
        return candidates


@pytest.fixture(scope="session")
def populate_databases() -> tuple[DummyParser, DummyParser]:
    parser1 = DummyParser(name="test_parser1", source="test_parser1")
    parser2 = DummyParser(name="test_parser2", source="test_parser2")
    for parser in [parser1, parser2]:
        parser.populate_databases(force=True)
    return parser1, parser2


class TestStrategy(MappingStrategy):
    """Implementation of MappingStrategy for testing."""

    def __init__(
        self,
        confidence: StringMatchConfidence,
        ent_match: str,
        expected_ids: set[str],
        disambiguation_strategies: Optional[list[DisambiguationStrategy]] = None,
    ):
        """

        :param confidence:
        :param ent_match: filter_terms only fires when ent_match is this this value
        :param expected_ids: only return CandidatesToMetrics which have this id
        :param disambiguation_strategies:
        """
        super().__init__(confidence, disambiguation_strategies)
        # note, we assign to expected_id in the .prepare method, so we can check .prepare is being called properly too
        self.expected_ids: set[str] = set()
        self.temp_ids = expected_ids
        self.ent_match = ent_match

    def prepare(self, document: Document):
        self.expected_ids = self.temp_ids

    def filter_candidates(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        candidates: CandidatesToMetrics,
        parser_name: str,
    ) -> CandidatesToMetrics:
        if ent_match == self.ent_match:
            return {
                candidate: metrics
                for candidate, metrics in candidates.items()
                for id_set in candidate.associated_id_sets
                if any(expected_id in id_set.ids for expected_id in self.expected_ids)
            }
        else:
            return {}


class TestDoNothingDisambiguationStrategy(DisambiguationStrategy):
    """Implementation of DisambiguationStrategy for testing."""

    def prepare(self, document: Document):
        pass

    def disambiguate(
        self,
        id_sets: set[EquivalentIdSet],
        document: Document,
        parser_name: str,
        ent_match: Optional[str] = None,
        ent_match_norm: Optional[str] = None,
    ) -> set[EquivalentIdSet]:
        return set()


class TestDisambiguationStrategy(DisambiguationStrategy):
    def __init__(self, expected_id: str, confidence: DisambiguationConfidence):
        # as above, we check .prepare is being called properly in DisambiguationStrategy by assigning the intended
        # value in .prepare
        super().__init__(confidence)
        self.expected_id: str = ""
        self.temp_id = expected_id

    def prepare(self, document: Document):
        self.expected_id = self.temp_id

    def disambiguate(
        self,
        id_sets: set[EquivalentIdSet],
        document: Document,
        parser_name: str,
        ent_match: Optional[str] = None,
        ent_match_norm: Optional[str] = None,
    ) -> set[EquivalentIdSet]:
        return set(id_set for id_set in id_sets if self.expected_id in id_set.ids)


def build_runner(expected_id_groups: dict[str, set[str]]) -> StrategyRunner:
    """Create a StrategyRunner configured with dummy strategies for testing.

    :param expected_id_groups:
    :return:
    """
    first_test_strategy = TestStrategy(
        StringMatchConfidence.HIGHLY_LIKELY,
        ent_match="test_1",
        expected_ids=expected_id_groups["test_1"],
    )
    second_test_strategy = TestStrategy(
        StringMatchConfidence.HIGHLY_LIKELY,
        ent_match="test_2",
        expected_ids=expected_id_groups["test_2"],
    )
    third_test_strategy = TestStrategy(
        StringMatchConfidence.HIGHLY_LIKELY,
        ent_match="test_3",
        expected_ids=expected_id_groups["test_3"],
    )
    fourth_test_strategy = TestStrategy(
        StringMatchConfidence.HIGHLY_LIKELY,
        ent_match="test_4",
        expected_ids=expected_id_groups["test_4"],
        disambiguation_strategies=[
            TestDoNothingDisambiguationStrategy(DisambiguationConfidence.AMBIGUOUS),
            TestDisambiguationStrategy(
                expected_id=next(iter(expected_id_groups["test_4"])),
                confidence=DisambiguationConfidence.POSSIBLE,
            ),
        ],
    )
    fifth_test_strategy = NoopMappingStrategy(StringMatchConfidence.HIGHLY_LIKELY)

    default_strategy = TestStrategy(
        StringMatchConfidence.HIGHLY_LIKELY,
        ent_match="unknown",
        expected_ids=expected_id_groups["test_default"],
    )

    strategy_execution = ConfidenceLevelStrategyExecution(
        ent_class_strategies={
            "test_class": [
                first_test_strategy,
                second_test_strategy,
                third_test_strategy,
                fourth_test_strategy,
                fifth_test_strategy,
            ]
        },
        default_strategies=[default_strategy],
    )
    symbolic_strategies = {
        MentionConfidence.HIGHLY_LIKELY.name: strategy_execution,
        MentionConfidence.PROBABLE.name: strategy_execution,
    }
    non_symbolic_strategies = symbolic_strategies

    runner = StrategyRunner(
        symbolic_strategies=symbolic_strategies, non_symbolic_strategies=non_symbolic_strategies
    )
    return runner


def create_test_doc(
    parser1_hit_1: LinkingCandidate,
    parser1_hit_2: LinkingCandidate,
    parser2_hit_1: LinkingCandidate,
    parser2_hit_2: LinkingCandidate,
) -> tuple[Document, dict[str, list[Entity]]]:
    """Create a test doc with dummy entities.

    :param parser1_hit_1:
    :param parser1_hit_2:
    :param parser2_hit_1:
    :param parser2_hit_2:
    :return: Document and dict of <test_name_key>:<list of entities associated with this
        test>
    """

    #
    e1_group_1 = Entity.load_contiguous_entity(
        start=0,
        end=1,
        match="test_1",
        entity_class="test_class",
        namespace="group1",
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
    )
    e1_group_1.add_or_update_linking_candidate(parser1_hit_1, LinkingMetrics())
    e2_group_1 = Entity.load_contiguous_entity(
        start=10,
        end=12,
        match="test_1",
        entity_class="test_class",
        namespace="group1",
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
    )
    e2_group_1.add_or_update_linking_candidate(parser1_hit_1, LinkingMetrics())
    e1_group_2 = Entity.load_contiguous_entity(
        start=0,
        end=1,
        match="test_2",
        entity_class="test_class",
        namespace="group2",
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
    )
    e1_group_2.add_or_update_linking_candidate(parser2_hit_1, LinkingMetrics())
    e2_group_2 = Entity.load_contiguous_entity(
        start=15,
        end=20,
        match="test_2",
        entity_class="test_class",
        namespace="group2",
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
    )
    e2_group_2.add_or_update_linking_candidate(parser2_hit_1, LinkingMetrics())

    e1_group_3 = Entity.load_contiguous_entity(
        start=0,
        end=1,
        match="test_3",
        entity_class="test_class",
        namespace="group3",
        mention_confidence=MentionConfidence.PROBABLE,
    )
    e1_group_3.add_or_update_linking_candidates(
        {parser1_hit_1: LinkingMetrics(), parser2_hit_1: LinkingMetrics()}
    )
    e2_group_3 = Entity.load_contiguous_entity(
        start=15,
        end=20,
        match="test_3",
        entity_class="test_class",
        namespace="group3",
        mention_confidence=MentionConfidence.PROBABLE,
    )
    e2_group_3.add_or_update_linking_candidates(
        {parser1_hit_1: LinkingMetrics(), parser2_hit_1: LinkingMetrics()}
    )

    e1_group_4 = Entity.load_contiguous_entity(
        start=0,
        end=1,
        match="test_4",
        entity_class="test_class",
        namespace="group4",
        mention_confidence=MentionConfidence.PROBABLE,
    )
    e1_group_4.add_or_update_linking_candidates(
        {parser1_hit_1: LinkingMetrics(), parser1_hit_2: LinkingMetrics()}
    )
    e2_group_4 = Entity.load_contiguous_entity(
        start=15,
        end=20,
        match="test_4",
        entity_class="test_class",
        namespace="group4",
        mention_confidence=MentionConfidence.PROBABLE,
    )
    e2_group_4.add_or_update_linking_candidates(
        {parser1_hit_1: LinkingMetrics(), parser1_hit_2: LinkingMetrics()}
    )

    e1_group_5 = Entity.load_contiguous_entity(
        start=0,
        end=1,
        match="test_5",
        entity_class="test_class",
        namespace="group5",
        mention_confidence=MentionConfidence.PROBABLE,
    )
    e1_group_5.add_or_update_linking_candidates(
        {parser2_hit_1: LinkingMetrics(), parser2_hit_2: LinkingMetrics()}
    )
    e2_group_5 = Entity.load_contiguous_entity(
        start=15,
        end=20,
        match="test_5",
        entity_class="test_class",
        namespace="group5",
        mention_confidence=MentionConfidence.PROBABLE,
    )
    e2_group_5.add_or_update_linking_candidates(
        {parser2_hit_1: LinkingMetrics(), parser2_hit_2: LinkingMetrics()}
    )

    e1_group_default = Entity.load_contiguous_entity(
        start=0,
        end=1,
        match="test_default",
        entity_class="unknown",
        namespace="group_default",
        mention_confidence=MentionConfidence.PROBABLE,
    )
    e1_group_default.add_or_update_linking_candidates(
        {parser1_hit_2: LinkingMetrics(), parser2_hit_2: LinkingMetrics()}
    )
    e2_group_default = Entity.load_contiguous_entity(
        start=15,
        end=20,
        match="test_default",
        entity_class="unknown",
        namespace="group_default",
        mention_confidence=MentionConfidence.PROBABLE,
    )
    e2_group_default.add_or_update_linking_candidates(
        {parser1_hit_2: LinkingMetrics(), parser2_hit_2: LinkingMetrics()}
    )

    doc = Document.create_simple_document("hello")
    doc.sections[0].entities = [
        e1_group_1,
        e2_group_1,
        e1_group_2,
        e2_group_2,
        e1_group_3,
        e2_group_3,
        e1_group_4,
        e2_group_4,
        e1_group_5,
        e2_group_5,
        e1_group_default,
        e2_group_default,
    ]

    return doc, {
        "test_1": [e1_group_1, e2_group_1],
        "test_2": [e1_group_2, e2_group_2],
        "test_3": [e1_group_3, e2_group_3],
        "test_4": [e1_group_4, e2_group_4],
        "test_5": [e1_group_5, e2_group_5],
        "test_default": [e1_group_3, e2_group_3],
    }


def build_and_execute_runner(
    populate_databases: tuple[DummyParser, DummyParser],
) -> tuple[dict[str, set[str]], dict[str, list[Entity]]]:
    parser1, parser2 = populate_databases
    expected_id_groups = extract_expected_ids_from_parsers(parser1, parser2)
    parser1_hit_1, parser1_hit_2, parser2_hit_1, parser2_hit_2 = extract_terms_from_parsers(
        parser1, parser2
    )
    doc, test_groups = create_test_doc(parser1_hit_1, parser1_hit_2, parser2_hit_1, parser2_hit_2)
    runner = build_runner(expected_id_groups)
    runner(doc)
    return expected_id_groups, test_groups


def extract_expected_ids_from_parsers(
    parser1: DummyParser, parser2: DummyParser
) -> dict[str, set[str]]:
    expected_idx_1 = parser1.data[IDX][0]
    assert expected_idx_1 == "first"
    expected_idx_2 = parser2.data[IDX][2]
    assert expected_idx_2 == "second"
    expected_idx_3 = parser2.data[IDX][4]
    assert expected_idx_3 == "third"

    return {
        "test_1": {expected_idx_1},
        "test_2": {expected_idx_2},
        "test_3": {expected_idx_1, expected_idx_2},
        "test_4": {expected_idx_3},
        "test_5": {expected_idx_2, expected_idx_3},
        "test_default": {expected_idx_1, expected_idx_2},
    }


def extract_terms_from_parsers(
    parser1: DummyParser, parser2: DummyParser
) -> tuple[LinkingCandidate, LinkingCandidate, LinkingCandidate, LinkingCandidate]:
    """Extract a tuple of Linking Candidates we need for testing.

    :param parser1:
    :param parser2:
    :return:
    """
    parser1_hit_1 = SynonymDatabase().get(parser1.name, synonym=parser1.data[SYN][0])

    parser1_hit_2 = SynonymDatabase().get(parser1.name, synonym=parser1.data[SYN][4])

    parser2_hit_1 = SynonymDatabase().get(parser2.name, synonym=parser2.data[SYN][2])

    parser2_hit_2 = SynonymDatabase().get(parser2.name, synonym=parser2.data[SYN][4])

    return parser1_hit_1, parser1_hit_2, parser2_hit_1, parser2_hit_2


def check_mappings_from_ents(
    ents: list[Entity], expected_mapping_count: int, expected_ids: set[str], message: str
):

    for ent in ents:
        assert len(ent.mappings) == expected_mapping_count, message
        for mapping in ent.mappings:
            expected_ids.discard(mapping.idx)

    assert not expected_ids, message


def test_StrategyRunner(mock_kazu_disk_cache_on_parsers, populate_databases):
    expected_id_groups, test_groups = build_and_execute_runner(populate_databases)
    message = (
        "first test: the entity has a single term from a single parser."
        " It should receive mapping with group_1_expected_idx (first configured strategy fires)"
    )
    check_mappings_from_ents(test_groups["test_1"], 1, expected_id_groups["test_1"], message)
    message = (
        "second test: same as group 1, except the first strategy should fail, and the second strategy should "
        "succeed. It should receive mapping with group_1_expected_idx (second strategy fires)"
    )
    check_mappings_from_ents(test_groups["test_2"], 1, expected_id_groups["test_2"], message)
    message = (
        "third test: the entities have a single term from multiple parser namespace. Should receive a mapping "
        "from each"
    )
    check_mappings_from_ents(test_groups["test_3"], 2, expected_id_groups["test_3"], message)
    message = (
        "fourth test: the entities have multiple terms from a single parser namespace. A disambiguation strategy "
        "is configured, so only one mapping is produced"
    )
    check_mappings_from_ents(test_groups["test_4"], 1, expected_id_groups["test_4"], message)
    message = (
        "fifth test: the entities have multiple terms from a parser namespace. no disambiguation strategy is "
        "configured, so multiple mappings expected"
    )
    check_mappings_from_ents(test_groups["test_5"], 2, expected_id_groups["test_5"], message)

    message = (
        "default test: the entities have a single term from multiple parser namespaces. all default strategies should fail,"
        "# so the runner uses the default strategy, which succeeds, producing a mapping from each namespace "
    )
    check_mappings_from_ents(
        test_groups["test_default"], 2, expected_id_groups["test_default"], message
    )
