import logging
from typing import List, Optional

from azner.data.data import (
    TokenizedWord,
    ENTITY_START_SYMBOL,
    ENTITY_INSIDE_SYMBOL,
    ENTITY_OUTSIDE_SYMBOL,
)

logger = logging.getLogger(__name__)


class BioLabelPreProcessor:
    """
    because of the inherent obscurity of the inner workings of transformers, sometimes they produce BIO tags that
    don't correctly align to whole words. So what do we do? Fix it with rules :~Z

    This class is designed to work when an entire sequence of NER labels is known and therefore we can apply some
    post-processing logic - i.e. a hack until we have time to debug the model
    """

    def test_for_single_token(self, word: TokenizedWord) -> bool:
        """
        single token word - do nothing
        :param word:
        :return:
        """
        return len(word.word_labels) == 1

    def _test_for_perfectly_formed_bio_word(self, word: TokenizedWord) -> bool:
        """
        ideal scenario = BIO is perfectly formed, or all labels are outside
        :param word:
        :return:
        """
        maybe_begin = word.bio_labels[0]
        maybe_insides = word.bio_labels[1:]
        return (
            maybe_begin == ENTITY_START_SYMBOL
            and len(set(maybe_insides)) == 1
            and maybe_insides[0] == ENTITY_INSIDE_SYMBOL
        ) or all([x == ENTITY_OUTSIDE_SYMBOL for x in word.bio_labels])

    def _test_for_b_then_mixed_bio(self, word: TokenizedWord) -> bool:
        """
        does the word start with a B, but contain a mix of other bio instead of I's?
        :param word:
        :return:
        """
        maybe_begin = word.bio_labels[0]
        maybe_insides = word.bio_labels[1:]
        if maybe_begin == ENTITY_START_SYMBOL and (
            (len(set(maybe_insides)) > 1) or (ENTITY_INSIDE_SYMBOL not in maybe_insides)
        ):
            logger.debug(f"result is malformed - reformatting labels to class of B: {maybe_begin}")
            return True
        else:
            return False

    def _fix_b_then_mixed_bio(self, word: TokenizedWord) -> None:
        """
        begin symbol detected, but I labels are broken. Convert everything after B to I's
        also:

        Since we're missing the I labels for this word, the confidences are probably off too.
        Therefore, we simply propagate the confidence from B label across all tokens, to give a consistent
        measure of confidence
        :param word:
        :return:
        """
        word.word_labels_strings = [f"{ENTITY_START_SYMBOL}-{word.class_labels[0]}"] + [
            f"{ENTITY_INSIDE_SYMBOL}-{word.class_labels[0]}"
            for _ in range(len(word.word_labels) - 1)
        ]

        b_confidence = word.word_confidences[0]
        word.word_confidences = [b_confidence for _ in range(len(word.word_confidences))]
        word.modified_post_inference = True

    def _test_for_missing_b_then_i(self, word: TokenizedWord, prev_word: TokenizedWord) -> bool:
        """
        sometimes, a B is missing from the start. Here, we need to check we're no inside a multi word entity
        i.e. last token of prev word is not I and class of last token of prev word is not same as word
        :param word:
        :return:
        """

        return (
            all([x == ENTITY_INSIDE_SYMBOL for x in word.bio_labels])
            and len(set(word.class_labels)) == 1
            and (
                (prev_word is None)
                or (prev_word.class_labels[-1] != word.class_labels[0])
                or (
                    prev_word.bio_labels[-1] != ENTITY_INSIDE_SYMBOL
                    and prev_word.class_labels[-1] != word.class_labels[0]
                )
            )
        )

    def _fix_missing_b_then_i(self, word: TokenizedWord) -> None:
        """
        convert first label to B-<class>
        :param word:
        :return:
        """
        word.word_labels_strings[0] = f"{ENTITY_START_SYMBOL}-{word.class_labels[0]}"
        word.modified_post_inference = True

    def _test_for_previous_word_is_i(self, word: TokenizedWord, prev_word: TokenizedWord) -> bool:
        """
        check if should be all I's as previous word is I
        :param word:
        :return:
        """

        return (
            all([x == ENTITY_INSIDE_SYMBOL for x in word.bio_labels])
            and len(set(word.class_labels)) == 1
            and (
                (prev_word is None)
                or (
                    prev_word.bio_labels[-1] == ENTITY_INSIDE_SYMBOL
                    and prev_word.class_labels[-1] == word.class_labels[0]
                )
            )
        )

    def fix_words(
        self, word: TokenizedWord, previous_word: Optional[TokenizedWord]
    ) -> TokenizedWord:
        """
        check if a word needs modification:
        :param word:
        :param state:
        :return:
        """
        if self.test_for_single_token(word):
            pass
        elif self._test_for_perfectly_formed_bio_word(word):
            pass
        elif self._test_for_previous_word_is_i(word, prev_word=previous_word):
            pass
        elif self._test_for_b_then_mixed_bio(word):
            self._fix_b_then_mixed_bio(word)
        elif self._test_for_missing_b_then_i(word, prev_word=previous_word):
            self._fix_missing_b_then_i(word)
        else:
            logger.warning(f"result is malformed for {word} - needs further postprocessing ideas")
        return word

    def __call__(self, words: List[TokenizedWord]) -> List[TokenizedWord]:
        [word.parse_labels_to_bio_and_class() for word in words]
        result = []
        for i, word in enumerate(words):
            if i == 0:
                result.append(self.fix_words(word, None))
            else:
                result.append(self.fix_words(word, words[i - 1]))
        return result
