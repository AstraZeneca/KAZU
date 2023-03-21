from kazu.tests.utils import DummyParser
from kazu.utils.link_index import DictionaryIndex


def test_DictionaryIndex():
    parser = DummyParser()
    index = DictionaryIndex(parser)
    terms = list(index.search("3"))
    assert len(terms) == 1
    term = terms[0]
    assert term.parser_name == parser.name
    assert term.exact_match

    terms = list(index.search("nothing"))

    assert all(term.search_score == 0.0 for term in terms)
