import logging

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

    def __call__(self, word: TokenizedWord) -> TokenizedWord:
        """
        note - this is a mutable call. Takes a TokenizedWord and alters labels according to a set of rules:
        :param word:
        :return:
        """
        if len(word.word_labels) == 1:
            # single token word - do nothing
            pass
        else:
            # we need to check everything is ok...
            word.parse_labels_to_bio_and_class()
            maybe_begin = word.bio_labels[0]
            maybe_insides = word.bio_labels[1:]
            if (
                maybe_begin == ENTITY_START_SYMBOL
                and len(set(maybe_insides)) == 1
                and maybe_insides[0] == ENTITY_INSIDE_SYMBOL
            ) or all([x == ENTITY_OUTSIDE_SYMBOL for x in word.bio_labels]):
                # ideal scenario = BIO is perfectly formed, or all labels are outside
                pass
            elif maybe_begin == ENTITY_START_SYMBOL:
                # begin symbol detected, but I labels are missing
                logger.debug(
                    f"result is malformed - reformatting labels to class of B: {maybe_begin}"
                )
                word.word_labels_strings = [f"{ENTITY_START_SYMBOL}-{word.class_labels[0]}"] + [
                    f"{ENTITY_INSIDE_SYMBOL}-{word.class_labels[0]}"
                    for _ in range(len(word.word_labels) - 1)
                ]
                # Since we're missing the I labels for this word, the confidences are probably off too.
                # Therefore, we simply propagate the confidence from B label across all tokens, to give a consistent
                # measure of confidence
                b_confidence = word.word_confidences[0]
                word.word_confidences = [b_confidence for _ in range(len(word.word_confidences))]
                word.modified_post_inference = True
            elif (
                all([x == ENTITY_INSIDE_SYMBOL for x in word.bio_labels])
                and len(set(word.class_labels)) == 1
            ):
                # word has all insides and same class label, but no begin. Therefore change first label to B
                word.word_labels_strings[0] = f"{ENTITY_START_SYMBOL}-{word.class_labels[0]}"
                word.modified_post_inference = True
            else:
                logger.warning(
                    f"result is malformed for {word} - needs further postprocessing ideas"
                )
        return word
