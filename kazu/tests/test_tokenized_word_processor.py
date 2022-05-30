import pytest
import torch

from kazu.steps.ner.tokenized_word_processor import TokenizedWord, TokenizedWordProcessor


def test_tokenized_word_processor_with_subspan_detection():
    text = "hello to you"
    # should produce one ent
    word1 = TokenizedWord(
        word_id=0,
        token_ids=[0],
        tokens=["hello"],
        token_confidences=[torch.Tensor([0.99, 0.01])],
        token_offsets=[(0, 5)],
        word_char_start=0,
        word_char_end=5,
    )
    word2 = TokenizedWord(
        word_id=1,
        token_ids=[1],
        tokens=["to"],
        token_confidences=[torch.Tensor([0.01, 0.99])],
        token_offsets=[(6, 8)],
        word_char_start=6,
        word_char_end=8,
    )
    word3 = TokenizedWord(
        word_id=2,
        token_ids=[2],
        tokens=["you"],
        token_confidences=[torch.Tensor([0.01, 0.99])],
        token_offsets=[(9, 11)],
        word_char_start=9,
        word_char_end=11,
    )

    processor = TokenizedWordProcessor(
        confidence_threshold=0.2, id2label={0: "B-hello", 1: "O"}, detect_subspans=True
    )
    ents = processor(words=[word1, word2, word3], text=text, namespace="test")
    assert len(ents) == 1
    assert ents[0].match == "hello"


def test_tokenized_word_processor_with_subspan_detection_2():
    text = "hello to you"
    # also check this works if the word hello is composed of two B tokens
    word1 = TokenizedWord(
        word_id=0,
        token_ids=[0, 1],
        tokens=["hel", "lo"],
        token_confidences=[torch.Tensor([0.99, 0.01]), torch.Tensor([0.99, 0.01])],
        token_offsets=[(0, 3), (3, 5)],
        word_char_start=0,
        word_char_end=5,
    )
    word2 = TokenizedWord(
        word_id=1,
        token_ids=[2],
        tokens=["to"],
        token_confidences=[torch.Tensor([0.01, 0.99])],
        token_offsets=[(6, 8)],
        word_char_start=6,
        word_char_end=8,
    )
    word3 = TokenizedWord(
        word_id=2,
        token_ids=[3],
        tokens=["you"],
        token_confidences=[torch.Tensor([0.01, 0.99])],
        token_offsets=[(9, 11)],
        word_char_start=9,
        word_char_end=11,
    )
    processor = TokenizedWordProcessor(
        confidence_threshold=0.2, id2label={0: "B-hello", 1: "O"}, detect_subspans=True
    )
    ents = processor(words=[word1, word2, word3], text=text, namespace="test")
    assert len(ents) == 1
    assert ents[0].match == "hello"


@pytest.mark.parametrize(
    "detect_subspans",
    (True, False),
)
def test_tokenized_word_processor_with_subspan_detection_3(detect_subspans: bool):
    text = "hello-to you"
    word1 = TokenizedWord(
        word_id=0,
        token_ids=[0],
        tokens=["hello"],
        token_confidences=[torch.Tensor([0.99, 0.01])],
        token_offsets=[(0, 5)],
        word_char_start=0,
        word_char_end=5,
    )
    word2 = TokenizedWord(
        word_id=1,
        token_ids=[1],
        tokens=["-"],
        token_confidences=[torch.Tensor([0.01, 0.99])],
        token_offsets=[(5, 6)],
        word_char_start=5,
        word_char_end=6,
    )
    word3 = TokenizedWord(
        word_id=2,
        token_ids=[2],
        tokens=["to"],
        token_confidences=[torch.Tensor([0.99, 0.01])],
        token_offsets=[(6, 8)],
        word_char_start=6,
        word_char_end=8,
    )
    word4 = TokenizedWord(
        word_id=3,
        token_ids=[3],
        tokens=["you"],
        token_confidences=[torch.Tensor([0.01, 0.99])],
        token_offsets=[(9, 11)],
        word_char_start=9,
        word_char_end=11,
    )

    processor = TokenizedWordProcessor(
        confidence_threshold=0.2,
        id2label={0: "B-greeting", 1: "O"},
        detect_subspans=detect_subspans,
    )
    ents = processor(words=[word1, word2, word3, word4], text=text, namespace="test")
    if detect_subspans:
        # should produce three ents, since '-' is non breaking
        assert len(ents) == 3
        assert ents[0].match == "hello-"
        assert ents[0].entity_class == "greeting"
        assert ents[1].match == "hello-to"
        assert ents[1].entity_class == "greeting"
        assert ents[2].match == "to"
        assert ents[2].entity_class == "greeting"
    else:
        # should produce two ents, since '-' is non breaking
        assert len(ents) == 2
        assert ents[0].match == "hello-"
        assert ents[0].entity_class == "greeting"
        assert ents[1].match == "hello-to"
        assert ents[1].entity_class == "greeting"


@pytest.mark.parametrize(
    "detect_subspans",
    (True, False),
)
def test_tokenized_word_processor_with_subspan_detection_4(detect_subspans):
    # should produce two ent as " " is span breaking
    text = "hello to you"
    word1 = TokenizedWord(
        word_id=0,
        token_ids=[0],
        tokens=["hello"],
        token_confidences=[torch.Tensor([0.99, 0.01])],
        token_offsets=[(0, 5)],
        word_char_start=0,
        word_char_end=5,
    )
    word2 = TokenizedWord(
        word_id=1,
        token_ids=[1],
        tokens=["to"],
        token_confidences=[torch.Tensor([0.01, 0.99])],
        token_offsets=[(6, 8)],
        word_char_start=6,
        word_char_end=8,
    )
    word3 = TokenizedWord(
        word_id=2,
        token_ids=[2],
        tokens=["you"],
        token_confidences=[torch.Tensor([0.99, 0.01])],
        token_offsets=[(9, 11)],
        word_char_start=9,
        word_char_end=12,
    )

    processor = TokenizedWordProcessor(
        confidence_threshold=0.2, id2label={0: "B-hello", 1: "O"}, detect_subspans=detect_subspans
    )
    ents = processor(words=[word1, word2, word3], text=text, namespace="test")
    assert len(ents) == 2
    assert ents[0].match == "hello"
    assert ents[1].match == "you"
