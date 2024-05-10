import pytest
from gliner.modules.token_splitter import WhitespaceTokenSplitter
from hydra.utils import instantiate

from kazu.data import Document, Section
from kazu.tests.utils import (
    requires_model_pack,
    ner_long_document_test_cases,
    maybe_skip_experimental_tests,
)
from kazu.utils.utils import token_sliding_window, sort_then_group


@requires_model_pack
def test_TransformersModelForTokenClassificationNerStep(kazu_test_config, ner_simple_test_cases):
    # note, here we just test that the step is functional. Model performance is tested via an acceptance test
    step = instantiate(kazu_test_config.TransformersModelForTokenClassificationNerStep)
    processed, failures = step(ner_simple_test_cases)
    assert len(processed) == len(ner_simple_test_cases)
    assert len(failures) == 0


@requires_model_pack
def test_multilabel_transformer_token_classification(
    override_kazu_test_config, ner_simple_test_cases
):
    # note, here we just test that the step is functional. Model performance is tested via an acceptance test
    cfg = override_kazu_test_config(
        overrides=["TransformersModelForTokenClassificationNerStep=multilabel"],
    )
    step = instantiate(cfg.TransformersModelForTokenClassificationNerStep)
    processed, failures = step(ner_simple_test_cases)
    assert len(processed) == len(ner_simple_test_cases)
    assert len(failures) == 0


@maybe_skip_experimental_tests
@requires_model_pack
def test_GLINERStep_simplecases(gliner_step, ner_simple_test_cases):
    processed, failures = gliner_step(ner_simple_test_cases)
    assert len(processed) == len(ner_simple_test_cases)
    assert len(failures) == 0


@maybe_skip_experimental_tests
@requires_model_pack
def test_GLINERStep_majority_vote(gliner_step):
    drug_section = Section(text="abracodabravir is a drug", name="test1")
    gene_section1 = Section(text="abracodabravir is a gene", name="test2")
    gene_section2 = Section(text="abracodabravir is definitely a gene", name="test3")
    conflicted_doc = Document(sections=[drug_section, gene_section1, gene_section2])
    processed, failures = gliner_step([conflicted_doc])
    assert len(processed) == 1
    assert len(failures) == 0
    for ent_class, ents in sort_then_group(conflicted_doc.get_entities(), lambda x: x.entity_class):
        assert ent_class == "gene", "non-gene entity types detected"
        assert len(list(ents)) == 3


@maybe_skip_experimental_tests
@requires_model_pack
def test_GLINERStep_windowing(gliner_step):
    doc_string, mention_count, long_doc_ent_class = ner_long_document_test_cases()[0]
    long_docs = [Document.create_simple_document(doc_string)]
    processed, failures = gliner_step(long_docs)
    assert len(processed) == len(long_docs)
    assert len(failures) == 0

    entities_grouped = {
        spans: list(ents)
        for spans, ents in sort_then_group(long_docs[0].sections[0].entities, lambda x: x.spans)
    }
    assert len(entities_grouped) == mention_count
    for ent_list in entities_grouped.values():
        assert len(ent_list) == 1
        assert ent_list[0].entity_class == long_doc_ent_class


@pytest.mark.parametrize(
    argnames=("stride", "window_size", "expected_texts"),
    argvalues=(
        (2, 10, ["ab cd ef gh ij kl mn op qr st ", "qr st uv wx yx"]),
        (
            4,
            5,
            [
                "ab cd ef gh ij ",
                "cd ef gh ij kl ",
                "ef gh ij kl mn ",
                "gh ij kl mn op ",
                "ij kl mn op qr ",
                "kl mn op qr st ",
                "mn op qr st uv ",
                "op qr st uv wx ",
                "qr st uv wx yx",
            ],
        ),
    ),
)
def test_string_windowing(stride, window_size, expected_texts):
    alphabet = "ab cd ef gh ij kl mn op qr st uv wx yx"

    test_window = list(WhitespaceTokenSplitter()(alphabet))
    frames = token_sliding_window(
        test_window, window_size=window_size, stride=stride, text=alphabet
    )

    captured_text = ""
    for (text, start_at, end_at), expected_text in zip(frames, expected_texts, strict=True):
        captured_text += alphabet[start_at:end_at]
        assert text == expected_text
    assert captured_text == alphabet
