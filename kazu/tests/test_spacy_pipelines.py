from pathlib import Path
from unittest.mock import patch

from kazu.utils.spacy_pipeline import SpacyPipelines, basic_spacy_pipeline, BASIC_PIPELINE_NAME
from spacy.matcher import PhraseMatcher
from kazu.utils.utils import Singleton

SHORT_TEXT = "Some random text and a WEirdTokenDDDD"
MEDIUM_TEXT = "Another bit of random text and a WiererdTokenDDDD"
LONG_TEXT = (
    "Yet another piece of random text and a WigrardTokenDDDD. "
    "A long bit of text will result in a larger string store"
)


def test_string_store_is_dereferenced():
    Singleton.clear_all()
    spacy_pipelines = SpacyPipelines()
    spacy_pipelines.reload_at = 3
    spacy_pipelines.add_from_func(BASIC_PIPELINE_NAME, basic_spacy_pipeline)
    doc1 = spacy_pipelines.process_single(SHORT_TEXT, model_name=BASIC_PIPELINE_NAME)
    doc2 = spacy_pipelines.process_single(MEDIUM_TEXT, model_name=BASIC_PIPELINE_NAME)
    doc3 = spacy_pipelines.process_single(LONG_TEXT, model_name=BASIC_PIPELINE_NAME)
    # doc1 and doc2 share the same vocab object, as they use the same iteration of the spacy
    # model. doc3 does not share the vocab, as the model was reloaded for that call, and
    # therefore the vocab for doc1 and doc2 can safely be garbage collected when those
    # objects are dereferenced
    assert doc1.vocab is doc2.vocab
    assert doc1.vocab is not doc3.vocab


def test_reload_from_path(tmpdir):
    Singleton.clear_all()
    nlp1_path = Path(tmpdir) / "nlp1"
    spacy_pipelines = SpacyPipelines()
    spacy_pipelines.reload_at = 1
    nlp1 = basic_spacy_pipeline()
    nlp1.to_disk(nlp1_path)
    spacy_pipelines.add_from_path(BASIC_PIPELINE_NAME, str(nlp1_path))
    initial_string_store_size = len(spacy_pipelines.get_model(BASIC_PIPELINE_NAME).vocab.strings)
    spacy_pipelines.process_single(LONG_TEXT, model_name=BASIC_PIPELINE_NAME)
    string_store_size_1 = len(spacy_pipelines.get_model(BASIC_PIPELINE_NAME).vocab.strings)
    assert string_store_size_1 > initial_string_store_size
    spacy_pipelines.process_single(SHORT_TEXT, model_name=BASIC_PIPELINE_NAME)
    final_string_store_size = len(spacy_pipelines.get_model(BASIC_PIPELINE_NAME).vocab.strings)
    assert final_string_store_size < string_store_size_1


def test_reload_from_func():
    Singleton.clear_all()
    spacy_pipelines = SpacyPipelines()
    spacy_pipelines.reload_at = 2
    spacy_pipelines.add_from_func(BASIC_PIPELINE_NAME, basic_spacy_pipeline)
    initial_string_store_size = len(spacy_pipelines.get_model(BASIC_PIPELINE_NAME).vocab.strings)
    long_doc = spacy_pipelines.process_single(LONG_TEXT, model_name=BASIC_PIPELINE_NAME)
    string_store_size_1 = len(spacy_pipelines.get_model(BASIC_PIPELINE_NAME).vocab.strings)
    assert string_store_size_1 > initial_string_store_size
    short_doc = spacy_pipelines.process_single(SHORT_TEXT, model_name=BASIC_PIPELINE_NAME)
    final_string_store_size = len(spacy_pipelines.get_model(BASIC_PIPELINE_NAME).vocab.strings)
    assert final_string_store_size < string_store_size_1
    # check we can still reference the string stores in the doc object, even when
    # it's no longer held in the models
    assert long_doc.text == LONG_TEXT
    assert short_doc.text == SHORT_TEXT


A_STRANGE_TOKEN = "AStrangeToken"


class CallBackTest:
    def __init__(self):
        self.spacy_pipelines = SpacyPipelines()
        self.init_matcher()

    def add_rule(self):
        self.matcher.add(
            "strange_rule", [self.spacy_pipelines.get_model(BASIC_PIPELINE_NAME)(A_STRANGE_TOKEN)]
        )

    def init_matcher(self) -> None:
        self.matcher = PhraseMatcher(self.spacy_pipelines.get_model(BASIC_PIPELINE_NAME).vocab)


def test_reload_with_callbacks():
    Singleton.clear_all()
    spacy_pipelines = SpacyPipelines()
    spacy_pipelines.reload_at = 2
    spacy_pipelines.add_from_func(BASIC_PIPELINE_NAME, basic_spacy_pipeline)
    call_back_test = CallBackTest()
    with patch.object(CallBackTest, "init_matcher", wraps=call_back_test.init_matcher) as mock:
        call_back_test.init_matcher()
        spacy_pipelines.add_reload_callback_func(BASIC_PIPELINE_NAME, call_back_test.init_matcher)
        assert (
            A_STRANGE_TOKEN not in spacy_pipelines.get_model(BASIC_PIPELINE_NAME).vocab.strings
        )  # the vocab doesn't have A_STRANGE_TOKEN until we call 'add_rule'
        call_back_test.add_rule()
        assert A_STRANGE_TOKEN in spacy_pipelines.get_model(BASIC_PIPELINE_NAME).vocab.strings
        # after processing one document, the vocab should still have A_STRANGE_TOKEN
        spacy_pipelines.process_single(SHORT_TEXT, model_name=BASIC_PIPELINE_NAME)
        assert A_STRANGE_TOKEN in spacy_pipelines.get_model(BASIC_PIPELINE_NAME).vocab.strings
        # after processing a second document, the model should have been reloaded, and A_STRANGE_TOKEN
        # should no longer be in it
        spacy_pipelines.process_single(MEDIUM_TEXT, model_name=BASIC_PIPELINE_NAME)
        assert A_STRANGE_TOKEN not in spacy_pipelines.get_model(BASIC_PIPELINE_NAME).vocab.strings
        # check init_matcher was called twice
        assert mock.call_count == 2


def test_batch():
    Singleton.clear_all()
    spacy_pipelines = SpacyPipelines()
    spacy_pipelines.reload_at = 2
    spacy_pipelines.add_from_func(BASIC_PIPELINE_NAME, basic_spacy_pipeline)
    initial_vocab = spacy_pipelines.get_model(BASIC_PIPELINE_NAME).vocab
    list(
        spacy_pipelines.process_batch(
            [SHORT_TEXT, MEDIUM_TEXT, LONG_TEXT], model_name=BASIC_PIPELINE_NAME
        )
    )
    reloaded_vocab = spacy_pipelines.get_model(BASIC_PIPELINE_NAME).vocab
    list(
        spacy_pipelines.process_batch(
            [SHORT_TEXT, MEDIUM_TEXT, LONG_TEXT], model_name=BASIC_PIPELINE_NAME
        )
    )
    assert initial_vocab is not reloaded_vocab
