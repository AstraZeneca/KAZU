from collections import defaultdict
from typing import Dict, Set

import pytest

import spacy
from spacy.lang.en import English

from kazu.data.data import SynonymData, EquivalentIdAggregationStrategy
from kazu.modelling.ontology_matching.ontology_matcher import OntologyMatcher
from kazu.modelling.ontology_matching.assemble_pipeline import main as assemble_pipeline


def test_constructor():
    nlp = English()
    default_config = {
        "span_key": "my_results",
        "entry_filter": {"@misc": "arizona.entry_filter_blacklist.v1"},
        "variant_generator": {"@misc": "arizona.variant_generator.v1"},
        "parser_name_to_entity_type": {},
    }
    ontology_matcher = OntologyMatcher(nlp, **default_config)
    assert ontology_matcher.span_key == "my_results"
    assert ontology_matcher.nr_strict_rules == 0
    assert ontology_matcher.nr_lowercase_rules == 0
    assert ontology_matcher.labels == []


def test_initialize():
    nlp = English()
    config = {"parser_name_to_entity_type": {}}
    ontology_matcher = nlp.add_pipe("ontology_matcher", config=config)
    assert isinstance(ontology_matcher, OntologyMatcher)
    # no matcher rules are defined
    nlp.initialize()
    assert ontology_matcher.nr_strict_rules == 0
    assert ontology_matcher.nr_lowercase_rules == 0


class MockParser:
    def __init__(self, parser_name: str, id_to_syns: Dict[str, Set[str]]):
        self.name = parser_name
        self.syn_to_syn_data: Dict[str, Set[SynonymData]] = defaultdict(set)
        # format input into structure of the output of collect_aggregate_synonym_data
        # doing this lets us have a simpler structure for creating new parsers to extend
        # the tests if desired
        for id, syns in id_to_syns.items():
            for syn in syns:
                self.syn_to_syn_data[syn].add(
                    SynonymData(
                        aggregated_by=EquivalentIdAggregationStrategy.UNAMBIGUOUS,
                        ids=frozenset((id,)),
                    )
                )

    def collect_aggregate_synonym_data(
        self, normalise_original_syns: bool
    ) -> Dict[str, Set[SynonymData]]:
        return self.syn_to_syn_data

    def generate_synonyms(self):
        return {}


parser_1 = MockParser(
    "first_mock_parser", {"http://purl.obolibrary.org/obo/UBERON_042": {"Q42_label", "Q42_syn"}}
)


parser_2 = MockParser(
    "second_mock_parser", {"http://purl.obolibrary.org/obo/MONDO_08": {"Q8_label", "Q8_syn"}}
)

example_text = (
    "There is a Q42_ID and Q42_syn in this sentence, as well as Q42_syn & Q8_syn synonyms."
)


@pytest.mark.parametrize(
    ("labels", "match_len", "match_texts", "match_entity_classes", "match_kb_ids"),
    [
        ([], 0, set(), set(), set()),
        (
            ["ent_type_1"],
            2,
            {"Q42_syn"},
            {"ent_type_1"},
            {"http://purl.obolibrary.org/obo/UBERON_042"},
        ),
        (
            ["ent_type_2"],
            1,
            {"Q8_syn"},
            {"ent_type_2"},
            {"http://purl.obolibrary.org/obo/MONDO_08"},
        ),
        (
            ["ent_type_1", "ent_type_2"],
            3,
            {"Q42_syn", "Q8_syn"},
            {"ent_type_1", "ent_type_2"},
            {
                "http://purl.obolibrary.org/obo/UBERON_042",
                "http://purl.obolibrary.org/obo/MONDO_08",
            },
        ),
        (
            ["RANDOM LABEL", "ent_type_1", "ent_type_2"],
            3,
            {"Q42_syn", "Q8_syn"},
            {"ent_type_1", "ent_type_2"},
            {
                "http://purl.obolibrary.org/obo/UBERON_042",
                "http://purl.obolibrary.org/obo/MONDO_08",
            },
        ),
    ],
)
def test_results_and_serialization(
    tmp_path, labels, match_len, match_texts, match_entity_classes, match_kb_ids
):
    TEST_SPAN_KEY = "my_hits"
    TEST_OUTPUT_DIR = tmp_path / "ontology_pipeline"
    parser_name_to_entity_type = {
        parser_1.name: "ent_type_1",
        parser_2.name: "ent_type_2",
    }
    nlp = assemble_pipeline(
        parsers=[parser_1, parser_2],
        blacklisters={},
        parser_name_to_entity_type=parser_name_to_entity_type,
        labels=labels,
        output_dir=TEST_OUTPUT_DIR,
        span_key=TEST_SPAN_KEY,
    )

    doc = nlp(example_text)
    matches = doc.spans[TEST_SPAN_KEY]

    assert_matches(matches, match_len, match_texts, match_entity_classes, match_kb_ids)

    nlp2 = spacy.load(TEST_OUTPUT_DIR)

    doc2 = nlp2(example_text)
    matches2 = doc2.spans[TEST_SPAN_KEY]

    assert_matches(matches2, match_len, match_texts, match_entity_classes, match_kb_ids)

    assert set((m.start_char, m.end_char, m.text) for m in matches2) == set(
        (m.start_char, m.end_char, m.text) for m in matches
    )


def assert_matches(matches, match_len, match_texts, match_entity_classes, match_kb_ids):
    assert len(matches) == match_len
    assert set([m.text for m in matches]) == match_texts
    assert set([m.label_ for m in matches]) == match_entity_classes
    assert set([m.kb_id_ for m in matches]) == match_kb_ids
