from hydra.utils import instantiate

from kazu.data.data import SimpleDocument
from kazu.steps import SciSpacyAbbreviationExpansionStep


def test_abbreviation_expansion_non_destructive(kazu_test_config):
    step: SciSpacyAbbreviationExpansionStep = instantiate(
        kazu_test_config.SciSpacyAbbreviationExpansionStep
    )
    text = (
        "NSCLC (Non Small Cell Lung Cancer) is a form of cancer. EGFR (Epidermal Growth Factor Receptor) is "
        "a gene."
    )
    doc = SimpleDocument(text)
    success, failures = step([doc])
    section = success[0].sections[0]
    expanded_text = section.preprocessed_text
    abbreviations_mappings = section.offset_map
    for i, (modified_char_span, original_char_span) in enumerate(abbreviations_mappings.items()):
        if i == 0:
            assert (
                expanded_text[modified_char_span.start : modified_char_span.end]
                == "Epidermal Growth Factor Receptor"
            )
            assert text[original_char_span.start : original_char_span.end] == "EGFR"
        elif i == 1:
            assert (
                expanded_text[modified_char_span.start : modified_char_span.end]
                == "Non Small Cell Lung Cancer"
            )
            assert text[original_char_span.start : original_char_span.end] == "NSCLC"
