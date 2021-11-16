from typing import List

from data.data import TokenizedWord
from steps.ner.bio_label_preprocessor import BioLabelPreProcessor


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


def test_bio_label_preprocessor():
    preprocessor = BioLabelPreProcessor()
    # case 1: properly formed, no action required
    word_label_strings = ["B-gene", "I-gene", "I-gene"]
    expected_word_label_strings = ["B-gene", "I-gene", "I-gene"]
    word = make_tokenized_word(word_label_strings)
    word = preprocessor(word)

    for observed, expected in zip(word.word_labels_strings, expected_word_label_strings):
        assert observed == expected

    # case 2: all O, no action required
    word_label_strings = ["O", "O", "O"]
    expected_word_label_strings = ["O", "O", "O"]
    word = make_tokenized_word(word_label_strings)
    word = preprocessor(word)
    for observed, expected in zip(word.word_labels_strings, expected_word_label_strings):
        assert observed == expected

    # case 3: missing I's
    word_label_strings = ["B-gene", "O", "O"]
    expected_word_label_strings = ["B-gene", "I-gene", "I-gene"]
    word = make_tokenized_word(word_label_strings)
    word = preprocessor(word)
    for observed, expected in zip(word.word_labels_strings, expected_word_label_strings):
        assert observed == expected

    # case 4: missing B
    word_label_strings = ["I-gene", "I-gene", "I-gene"]
    expected_word_label_strings = ["B-gene", "I-gene", "I-gene"]
    word = make_tokenized_word(word_label_strings)
    word = preprocessor(word)
    for observed, expected in zip(word.word_labels_strings, expected_word_label_strings):
        assert observed == expected
