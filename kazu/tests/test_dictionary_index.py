import tempfile
from unittest.mock import patch

from kazu.modelling.ontology_preprocessing.base import OntologyParser
from kazu.tests.utils import DummyParser
from kazu.utils.link_index import DictionaryIndex
from kazu.utils.utils import get_cache_dir


def test_dictionary_index_caching():
    with tempfile.TemporaryDirectory("kazu") as f:
        parser = DummyParser(f)
        index = DictionaryIndex(DummyParser(f))
        cache_dir = get_cache_dir(
            f,
            prefix=f"{parser.name}_{index.__class__.__name__}",
            create_if_not_exist=False,
        )

        assert_cache_built(cache_dir, parser, f)
        asset_cache_loaded(cache_dir, parser, f)


def assert_search_is_working(parser: OntologyParser):
    index = DictionaryIndex(parser)
    index.load_or_build_cache(False)
    terms = list(index.search("3"))
    assert len(terms) == 1
    term = terms[0]
    assert term.parser_name == parser.name
    assert term.exact_match

    terms = list(index.search("nothing"))

    assert all(term.search_score == 0.0 for term in terms)


def asset_cache_loaded(cache_dir, parser, f):
    # now test that the prebuilt cache is loaded

    with patch("kazu.utils.link_index.DictionaryIndex.load") as load:
        index = DictionaryIndex(DummyParser(f))
        index.load_or_build_cache(False)
        load.assert_called_with(cache_dir)
    # now actually load the cache and check search is working
    assert_search_is_working(parser)


def assert_cache_built(cache_dir, parser, f):
    # test that the cache is built

    with patch(
        "kazu.utils.link_index.DictionaryIndex.build_ontology_cache"
    ) as build_ontology_cache:
        index = DictionaryIndex(DummyParser(f))
        index.load_or_build_cache(False)
        build_ontology_cache.assert_called_with(cache_dir)
    # now actually build the cache and check search is working
    assert_search_is_working(parser)
