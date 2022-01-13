import pytest

from kazu.data.data import TokenizedWord, SimpleDocument, CharSpan
from fastapi.encoders import jsonable_encoder


def test_tokenized_word():
    # fillers used to complete params, but not needed in tests
    word_label_strings = ["B-gene", "I-gene", "I-gene"]
    expected_bio_labels = ["B", "I", "I"]
    expected_class_labels = ["gene", "gene", "gene"]
    word_labels = [0 for _ in range(len(word_label_strings))]
    word_confidences = [0.5 for _ in range(len(word_label_strings))]
    word_offsets = [
        (
            0,
            1,
        )
        for _ in range(len(word_label_strings))
    ]
    word = TokenizedWord(
        word_labels=word_labels,
        word_labels_strings=word_label_strings,
        word_offsets=word_offsets,
        word_confidences=word_confidences,
    )
    assert len(word.bio_labels) == 0
    assert len(word.class_labels) == 0
    word.parse_labels_to_bio_and_class()
    assert len(word.bio_labels) == len(word_label_strings)
    assert len(word.class_labels) == len(word_label_strings)
    for expected, observed in zip(expected_bio_labels, word.bio_labels):
        assert expected == observed
    for expected, observed in zip(expected_class_labels, word.class_labels):
        assert expected == observed


def test_serialisation():
    x = SimpleDocument("Hello")
    x.sections[0].offset_map = {CharSpan(start=1, end=2): CharSpan(start=1, end=2)}
    # should raise a type error
    with pytest.raises(TypeError):
        jsonable_encoder(x)

    # call to as_serialisable means we can encode it
    y = x.as_serialisable()
    jsonable_encoder(y)
