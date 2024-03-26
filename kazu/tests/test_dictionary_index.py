import pytest
from kazu.tests.utils import DummyParser
from kazu.utils.link_index import DictionaryIndex

pytestmark = pytest.mark.usefixtures("mock_kazu_disk_cache_on_parsers")


def test_DictionaryIndex():
    parser = DummyParser()
    index = DictionaryIndex(parser)
    candidates_and_metrics = list(index.search("3"))
    assert len(candidates_and_metrics) == 1
    candidate, metrics = candidates_and_metrics[0]
    assert candidate.parser_name == parser.name
    assert metrics.exact_match

    candidates_and_metrics = list(index.search("nothing"))

    assert all(metrics.search_score == 0.0 for _, metrics in candidates_and_metrics)
