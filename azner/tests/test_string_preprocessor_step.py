from typing import List, Tuple, Optional

from azner.steps.string_preprocessing.string_preprocessing_step import StringPreprocessorStep
from data.data import Section, CharSpan, SimpleDocument


class AddSomeCharsStep(StringPreprocessorStep):
    def __init__(
        self,
        depends_on: Optional[List[str]],
        insert_string: str,
        insert_start: int,
        insert_end: int,
    ):
        super().__init__(depends_on)
        self.insert_start = insert_start
        self.insert_end = insert_end
        self.insert_string = insert_string

    def create_modifications(self, section: Section) -> List[Tuple[CharSpan, str]]:
        return [(CharSpan(start=self.insert_start, end=self.insert_end), self.insert_string)]


def test_multiple_string_preprocessing_steps():

    original_string_representation = "Hello"
    doc = SimpleDocument(original_string_representation)

    # case 1: overwrite original string with extra characters
    expansion_string_1 = "Hello look how I've grown! "
    step = AddSomeCharsStep([], expansion_string_1, 0, 5)
    success, _ = step([doc])
    section = success[0].sections[0]
    expected_string = "Hello look how I've grown! "
    assert_section_is_correct(
        expansion_string_1, original_string_representation, section, expected_string
    )

    # case 2: remove characters
    expansion_string_2 = "Hello again"
    step = AddSomeCharsStep([], expansion_string_2, 0, len(expansion_string_1))
    success, _ = step([doc])
    section = success[0].sections[0]
    expected_string = "Hello again"
    assert_section_is_correct(
        expansion_string_2, original_string_representation, section, expected_string
    )

    # case 3: substitute characters
    expansion_string_3 = "it's gone again!"
    step = AddSomeCharsStep([], expansion_string_3, 6, 11)
    success, _ = step([doc])
    section = success[0].sections[0]
    expected_string = "Hello it's gone again!"
    assert_section_is_correct(expansion_string_3, "", section, expected_string)

    # test case 1 again
    expansion_string_1 = "Hello look how I've grown! "
    step = AddSomeCharsStep([], expansion_string_1, 0, 5)
    success, _ = step([doc])
    section = success[0].sections[0]
    expected_string = "Hello look how I've grown!  it's gone again!"
    assert_section_is_correct(
        expansion_string_1, original_string_representation, section, expected_string
    )


def assert_section_is_correct(
    new_string: str, original_string_representation: str, section: Section, expected_string: str
):
    text = section.text
    expanded_text = section.preprocessed_text
    assert expanded_text == expected_string
    abbreviations_mappings = section.offset_map
    for i, (modified_char_span, original_char_span) in enumerate(abbreviations_mappings.items()):
        assert expanded_text[modified_char_span.start : modified_char_span.end] == new_string
        assert (
            text[original_char_span.start : original_char_span.end]
            == original_string_representation
        )
