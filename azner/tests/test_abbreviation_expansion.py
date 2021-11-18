from hydra import initialize_config_module, compose
from hydra.utils import instantiate

from azner.data.data import SimpleDocument
from azner.steps import SciSpacyAbbreviationExpansionStep


def test_abbreviation_expansion_non_destructive():
    with initialize_config_module(config_module="azner.conf"):
        cfg = compose(
            config_name="config",
            overrides=[],
        )

        step: SciSpacyAbbreviationExpansionStep = instantiate(cfg.SciSpacyAbbreviationExpansionStep)
        text = (
            "NSCLC (Non Small Cell Lung Cancer) is a form of cancer. EGFR (Epidermal Growth Factor Receptor) is "
            "a gene."
        )
        doc = SimpleDocument(text)
        success, failures = step([doc])
        section = success[0].sections[0]
        expanded_text = section.preprocessed_text
        abbreviations_mappings = section.offset_map
        for i, (modified_char_span, original_char_span) in enumerate(
            abbreviations_mappings.items()
        ):
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
