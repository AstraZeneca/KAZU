import pytest
import torch

from kazu.steps.ner.tokenized_word_processor import TokenizedWord, TokenizedWordProcessor


def test_tokenized_word_processor_single_label():
    text = "hello to you"
    # should produce one ent
    word1 = TokenizedWord(
        word_id=0,
        token_ids=[0],
        tokens=["hello"],
        token_confidences=torch.Tensor([[0.70, 0.20, 0.10]]),
        token_offsets=[(0, 5)],
        word_char_start=0,
        word_char_end=5,
    )
    word2 = TokenizedWord(
        word_id=1,
        token_ids=[1],
        tokens=["to"],
        token_confidences=torch.Tensor([[0.01, 0.98, 0.01]]),
        token_offsets=[(6, 8)],
        word_char_start=6,
        word_char_end=8,
    )
    word3 = TokenizedWord(
        word_id=2,
        token_ids=[2],
        tokens=["you"],
        token_confidences=torch.Tensor([[0.01, 0.01, 0.98]]),
        token_offsets=[(9, 11)],
        word_char_start=9,
        word_char_end=11,
    )

    processor = TokenizedWordProcessor(labels=["B-class1", "O", "B-class2"], use_multilabel=False)
    ents = processor(words=[word1, word2, word3], text=text, namespace="test")
    assert len(ents) == 2
    detected_ent_classes = [ent.entity_class for ent in ents]
    assert "class1" in detected_ent_classes
    assert "class2" in detected_ent_classes


def test_tokenized_word_processor_multi_label():
    text = "hello to you"
    # should produce one ent
    word1 = TokenizedWord(
        word_id=0,
        token_ids=[0],
        tokens=["hello"],
        token_confidences=torch.Tensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]),
        token_offsets=[(0, 5)],
        word_char_start=0,
        word_char_end=5,
    )
    word2 = TokenizedWord(
        word_id=1,
        token_ids=[1],
        tokens=["to"],
        token_confidences=torch.Tensor([[[1, 0, 0], [0, 0, 0], [0, 0, 1]]]),
        token_offsets=[(6, 8)],
        word_char_start=6,
        word_char_end=8,
    )
    word3 = TokenizedWord(
        word_id=2,
        token_ids=[2],
        tokens=["you"],
        token_confidences=torch.Tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
        token_offsets=[(9, 11)],
        word_char_start=9,
        word_char_end=11,
    )

    processor = TokenizedWordProcessor(labels=["class1", "O", "class2"], use_multilabel=True)
    ents = processor(words=[word1, word2, word3], text=text, namespace="test")
    assert len(ents) == 2
    detected_ent_classes = set()
    detected_ent_matches = set()
    for ent in ents:
        detected_ent_classes.add(ent.entity_class)
        detected_ent_matches.add(ent.match)
    assert "class1" in detected_ent_classes
    assert "class2" in detected_ent_classes
    assert "to" in detected_ent_matches
    assert "hello to" in detected_ent_matches


@pytest.mark.parametrize("query", ["COX2 protein", "COX2 gene", "COX2 gene protein protein gene"])
def test_tokenized_word_processor_strip_re(query):
    processor = TokenizedWordProcessor(
        labels=["B-hello", "O"], use_multilabel=False, strip_re={"gene": "( (gene|protein)s?)+$"}
    )
    expected_str = "COX2"
    expected_end = 4

    result_str, result_end = processor.attempt_strip_suffixes(
        start=0, end=len(query), match_str=query, clazz="gene"
    )
    assert result_str == expected_str and result_end == expected_end
    result_str, result_end = processor.attempt_strip_suffixes(
        start=0, end=len(query), match_str=query, clazz="none"
    )
    assert result_str == query and result_end == len(query)
