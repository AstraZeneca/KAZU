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
        expanded_text = section.metadata.get(step.PREPROCESSED_TEXT_KEY)
        abbreviations_mappings = section.metadata.get(step.OFFSET_MAP_KEY)
        for i, (key_tup, original_char_span) in enumerate(abbreviations_mappings.items()):
            ex_start, ex_end = key_tup
            if i == 0:
                assert expanded_text[ex_start:ex_end] == "Epidermal Growth Factor Receptor"
                assert text[original_char_span.start : original_char_span.end] == "EGFR"
            elif i == 1:
                assert expanded_text[ex_start:ex_end] == "Non Small Cell Lung Cancer"
                assert text[original_char_span.start : original_char_span.end] == "NSCLC"


def test_abbreviation_expansion_destructive():
    with initialize_config_module(config_module="azner.conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "SciSpacyAbbreviationExpansionStep.override_original_section_text=True",
            ],
        )

        step = instantiate(cfg.SciSpacyAbbreviationExpansionStep)
        text = (
            "NSCLC (Non Small Cell Lung Cancer) is a form of cancer. EGFR (Epidermal Growth Factor Receptor) "
            "is a gene."
        )
        expected_text = (
            "Non Small Cell Lung Cancer (Non Small Cell Lung Cancer) is a form of cancer. Epidermal "
            "Growth Factor Receptor (Epidermal Growth Factor Receptor) is a gene."
        )
        doc = SimpleDocument(text)
        success, failures = step([doc])
        section = success[0].sections[0]
        expanded_text = section.get_text()
        assert expanded_text == expected_text
        abbreviations_mappings = section.metadata.get("abbreviations_offset_mappings")
        for i, (key_tup, val_tup) in enumerate(abbreviations_mappings.items()):
            ex_start, ex_end = key_tup
            original_start, original_end = val_tup
            if i == 0:
                assert expanded_text[ex_start:ex_end] == "Epidermal Growth Factor Receptor"
                assert text[original_start:original_end] == "EGFR"
            elif i == 1:
                assert expanded_text[ex_start:ex_end] == "Non Small Cell Lung Cancer"
                assert text[original_start:original_end] == "NSCLC"
