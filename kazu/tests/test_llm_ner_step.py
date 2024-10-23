import json

import pytest

from kazu.data import Document, Section
from kazu.steps.ner.llm_ner import (
    LLMNERStep,
    LLMModel,
    LLM_RAW_RESPONSE,
    FreeFormResultParser,
    SectionProcessingStrategy,
)


class TestLLMModel(LLMModel):
    FAILED_RESPONSE = "llm failed to produce a json response"

    def __call__(self, text: str) -> str:
        if "gene" in text:
            return json.dumps({"EGFR": "gene"})
        elif "species" in text:
            return json.dumps([{"cat": "species"}])
        elif "blood test" in text:
            return json.dumps({"EGFR": "blood test"})
        else:
            return self.FAILED_RESPONSE


def _check_doc(doc: Document, expected_ents: int, expected_class: str):
    assert len(doc.get_entities()) == expected_ents
    assert all(ent.entity_class == expected_class for ent in doc.get_entities())


@pytest.mark.parametrize(
    "section_processing_strategy",
    [
        SectionProcessingStrategy.PROCESS_INDIVIDUALLY_AND_DROP_FAILED_SECTIONS,
        SectionProcessingStrategy.PROCESS_INDIVIDUALLY_AND_KEEP_FAILED_SECTIONS,
    ],
)
def test_llm_ner_step(section_processing_strategy):
    step = LLMNERStep(
        model=TestLLMModel(),
        result_parser=FreeFormResultParser(),
        section_processing_strategy=section_processing_strategy,
    )
    doc = Document.create_simple_document("EGFR is a gene. EGFR is a growth factor receptor.")
    # note, the below instance will be labelled as a gene. Although this is semantically
    # incorrect, we're testing the 'choose first class found, sequentially' logic of the step
    doc.sections.append(Section(text="EGFR could also be a blood test", name="test_section"))
    processed, failures = step([doc])
    assert len(processed) == 1
    assert len(failures) == 0
    _check_doc(processed[0], 3, "gene")
    doc = Document.create_simple_document("a cat is a species of animal.")
    processed, failures = step([doc])
    assert len(processed) == 1
    assert len(failures) == 0
    _check_doc(processed[0], 1, "species")
    doc = Document.create_simple_document("no entities here.")

    processed, failures = step([doc])
    if (
        section_processing_strategy
        is SectionProcessingStrategy.PROCESS_INDIVIDUALLY_AND_DROP_FAILED_SECTIONS
    ):
        assert len(processed) == 1
        assert len(failures) == 0
        assert len(processed[0].sections) == 0
    else:
        assert len(failures) == 1
        assert len(processed[0].sections[0].entities) == 0
        assert processed[0].sections[0].metadata[LLM_RAW_RESPONSE] == TestLLMModel.FAILED_RESPONSE
