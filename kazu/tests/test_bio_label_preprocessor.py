from typing import List

from kazu.data.data import TokenizedWord
from kazu.steps.ner.bio_label_preprocessor import BioLabelPreProcessor

preprocessor = BioLabelPreProcessor()


def make_tokenized_word(label_list: List[str]) -> TokenizedWord:
    word_labels = [0 for _ in range(len(label_list))]
    word_confidences = [0.5 for _ in range(len(label_list))]
    word_offsets = [
        (
            0,
            1,
        )
        for _ in range(len(label_list))
    ]
    word = TokenizedWord(
        word_labels=word_labels,
        word_labels_strings=label_list,
        word_offsets=word_offsets,
        word_confidences=word_confidences,
    )
    return word


def test_bio_label_preprocessor_1():

    # case 1: properly formed, no action required
    word_label_strings = ["B-gene", "I-gene", "I-gene"]
    expected_word_label_strings = ["B-gene", "I-gene", "I-gene"]
    word = make_tokenized_word(word_label_strings)
    words = preprocessor([word])
    for observed, expected in zip(words[0].word_labels_strings, expected_word_label_strings):
        assert observed == expected


def test_bio_label_preprocessor_2():
    # case 2: all O, no action required
    word_label_strings = ["O", "O", "O"]
    expected_word_label_strings = ["O", "O", "O"]
    word = make_tokenized_word(word_label_strings)
    words = preprocessor([word])
    for observed, expected in zip(words[0].word_labels_strings, expected_word_label_strings):
        assert observed == expected


def test_bio_label_preprocessor_3():
    # case 3: missing I's
    word_label_strings = ["B-gene", "O", "O"]
    expected_word_label_strings = ["B-gene", "I-gene", "I-gene"]
    word = make_tokenized_word(word_label_strings)
    words = preprocessor([word])
    for observed, expected in zip(words[0].word_labels_strings, expected_word_label_strings):
        assert observed == expected


def test_bio_label_preprocessor_4():
    # case 4: missing B
    word_label_strings = ["I-gene", "I-gene", "I-gene"]
    expected_word_label_strings = ["B-gene", "I-gene", "I-gene"]
    word = make_tokenized_word(word_label_strings)
    words = preprocessor([word])
    for observed, expected in zip(words[0].word_labels_strings, expected_word_label_strings):
        assert observed == expected


def test_bio_label_preprocessor_5():
    # case 5: multi word entity
    word1_label_strings = ["B-gene", "I-gene", "I-gene"]
    word2_label_strings = ["I-gene", "I-gene", "I-gene"]

    expected_word_label_strings = ["I-gene", "I-gene", "I-gene"]
    word1 = make_tokenized_word(word1_label_strings)
    word2 = make_tokenized_word(word2_label_strings)

    words = preprocessor([word1, word2])
    for observed, expected in zip(words[1].word_labels_strings, expected_word_label_strings):
        assert observed == expected


def test_bio_label_preprocessor_6():
    # case 6: two entities next to each other. Second needs processing
    word1_label_strings = ["B-disease", "I-disease", "I-disease"]
    word2_label_strings = ["I-gene", "I-gene", "I-gene"]

    expected_word_label_strings = ["B-gene", "I-gene", "I-gene"]
    word1 = make_tokenized_word(word1_label_strings)
    word2 = make_tokenized_word(word2_label_strings)

    words = preprocessor([word1, word2])
    for observed, expected in zip(words[1].word_labels_strings, expected_word_label_strings):
        assert observed == expected
