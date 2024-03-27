import pytest
from hydra.utils import instantiate

from kazu.data import Document
from kazu.tests.utils import requires_model_pack, ner_long_document_test_cases
from kazu.steps import Step


@requires_model_pack
def test_StanzaStep(kazu_test_config):
    step = instantiate(kazu_test_config.StanzaStep)
    docs = [Document.create_simple_document(x[0]) for x in ner_long_document_test_cases()]
    processed, failures = step(docs)
    assert len(processed) == len(docs) and len(failures) == 0


@requires_model_pack
def test_generates_correct_spans(kazu_test_config):
    step: Step = instantiate(kazu_test_config.StanzaStep)
    docs: list[Document] = [
        Document.create_simple_document(
            "this spans for char 0 to 28. This sentence spans from 29 to 63."
        )
    ]
    processed, failures = step(docs)
    assert len(processed) == len(docs) and len(failures) == 0
    sent_spans = list(processed[0].sections[0].sentence_spans)
    assert len(sent_spans) == 2
    assert sent_spans[0].start == 0 and sent_spans[0].end == 28
    assert sent_spans[1].start == 29 and sent_spans[1].end == 63


@pytest.mark.skip(
    reason="ExplosionStringMatchingStep is semi deprecated - not in default model pack so not built, making this test extrememly slow."
)
@requires_model_pack
def test_multiple_sentence_splitters_causes_error(kazu_test_config):
    st_step: Step = instantiate(kazu_test_config.StanzaStep)
    ex_step: Step = instantiate(
        kazu_test_config.ExplosionStringMatchingStep, include_sentence_offsets=True
    )

    docs: list[Document] = [
        Document.create_simple_document(
            "This spans from char 0 to 29. This sentence spans from 30 to 64."
        )
    ]

    st_step(docs)
    ex_processed, ex_failures = ex_step(docs)

    assert len(ex_failures) == 1


@pytest.mark.skip(
    reason="ExplosionStringMatchingStep is semi deprecated - not in default model pack so not built, making this test extrememly slow."
)
@requires_model_pack
def test_equivalent_to_explosion_for_simple_sents(kazu_test_config):
    st_step: Step = instantiate(kazu_test_config.StanzaStep)
    ex_step: Step = instantiate(
        kazu_test_config.ExplosionStringMatchingStep, include_sentence_offsets=True
    )

    docs: list[Document] = [
        Document.create_simple_document(
            "This spans from char 0 to 29. This sentence spans from 30 to 64."
        )
        for _ in range(2)
    ]

    st_processed, st_failures = st_step([docs[0]])
    ex_processed, ex_failures = ex_step([docs[1]])

    assert len(st_processed) == len(ex_processed) == 1 and len(st_failures) == len(ex_failures) == 0

    st_sent_spans = list(st_processed[0].sections[0].sentence_spans)
    ex_sent_spans = list(ex_processed[0].sections[0].sentence_spans)

    assert len(st_sent_spans) == len(ex_sent_spans) == 2

    assert (
        st_sent_spans[0].start == ex_sent_spans[0].start == 0
        and st_sent_spans[0].end == ex_sent_spans[0].end == 29
    )
    assert (
        st_sent_spans[1].start == ex_sent_spans[1].start == 30
        and st_sent_spans[1].end == ex_sent_spans[1].end == 64
    )
