from contextlib import nullcontext as does_not_raise
import tempfile
import traceback
from pathlib import Path

import pytest
from hydra.utils import instantiate

from kazu.data.data import Document, PROCESSING_EXCEPTION
from kazu.pipeline import FailedDocsFileHandler, Pipeline, PipelineValueError
from kazu.steps import Step, document_iterating_step
from kazu.tests.utils import requires_model_pack


class BrokenStep(Step):
    def __call__(self, docs: list[Document]) -> tuple[list[Document], list[Document]]:
        failed_docs = []
        for doc in docs:
            try:
                this_causes_an_exception = {{"cant": "hash a dict"}}
                this_causes_an_exception.clear()
            except Exception:
                doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                failed_docs.append(doc)
        return [], failed_docs


def test_pipeline_error_handling(tmp_path: Path):
    step = BrokenStep()
    pipeline = Pipeline([step], [FailedDocsFileHandler(tmp_path)])

    docs = [Document.create_simple_document("hello") for _ in range(5)]
    pipeline(docs)
    error_files = list(tmp_path.joinpath(step.namespace()).iterdir())
    # should be two files per doc - one with exception, one with doc contents
    assert len(error_files) == 2 * len(docs)

    # should flush docs between calls
    assert len(pipeline.failed_docs) == 0

    more_docs = [Document.create_simple_document("hello") for _ in range(5)]

    pipeline(more_docs)
    error_files = list(tmp_path.joinpath(step.namespace()).iterdir())
    # should be two files per doc - one with exception, one with doc contents
    assert len(error_files) == 2 * (len(docs) + len(more_docs))


@requires_model_pack
def test_full_pipeline_and_serialisation(kazu_test_config):
    # test the default pipeline can load/configs are all correct
    pipeline: Pipeline = instantiate(kazu_test_config.Pipeline)
    doc = Document.create_simple_document("EGFR is an important gene in breast cancer")
    doc = pipeline([doc])[0]
    with tempfile.TemporaryFile(mode="w") as f:
        f.write(doc.to_json())


class MetadataTaggingStep(Step):
    def __init__(self, tag: str):
        self.tag = tag

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        doc.metadata.setdefault("MetadataTaggingStepsRun", []).append(self.tag)

    def namespace(self) -> str:  # type:ignore[override]
        # type ignore because for the base class this is a classmethod
        # keeping this a classmethod would make the test setup more complex
        # for no real benefit here.
        return self.tag


ALL_TAGS = [f"Step{n}" for n in range(10)]  # note that there is a Step0
SUBSET_OF_TAGS = [ALL_TAGS[i] for i in (1, 3, 4, 6, 7)]
REVERSED_TAG_SUBSET = list(reversed(SUBSET_OF_TAGS))


@pytest.fixture(scope="module")
def metadata_tagging_pipeline():
    return Pipeline(
        steps=[MetadataTaggingStep(tag=t) for t in ALL_TAGS],
        step_groups={
            "group_with_one_step": ["Step5"],
            "group_with_several_steps": SUBSET_OF_TAGS,
        },
    )


@pytest.mark.parametrize(
    argnames=("pipeline_call_kwargs", "raising_context", "expected_tagging_steps_run"),
    argvalues=(
        # this first case really tests that MetadataTaggingSteps work the way we designed - testing the test code!
        pytest.param(
            {"step_namespaces": ALL_TAGS}, does_not_raise(), ALL_TAGS, id="test all steps"
        ),
        pytest.param(
            {"step_group": "group_with_one_step"},
            does_not_raise(),
            ["Step5"],
            id="test group with one step",
        ),
        pytest.param(
            {"step_group": "group_with_several_steps"},
            does_not_raise(),
            SUBSET_OF_TAGS,
            id="test group with several steps",
        ),
        pytest.param(
            {"step_namespaces": ["Step3"]},
            does_not_raise(),
            ["Step3"],
            id="test single step namespace",
        ),
        pytest.param(
            {"step_namespaces": REVERSED_TAG_SUBSET},
            does_not_raise(),
            REVERSED_TAG_SUBSET,
            id="test multiple namespaces, reversing order",
        ),
        pytest.param(
            {"step_namespaces": ["Step1", "Step2"], "step_group": "group_with_one_step"},
            pytest.raises(PipelineValueError),
            None,
            id="cant provide both namespaces and group",
        ),
        pytest.param(
            {"step_namespaces": ["my_fake_step"]},
            pytest.raises(PipelineValueError),
            None,
            id="namespace provided that doesnt exist",
        ),
        pytest.param(
            {"step_group": "my_fake_group"},
            pytest.raises(PipelineValueError),
            None,
            id="step group provided that doesnt exist",
        ),
    ),
)
def test_invalid_pipeline_call_arguments_raise_as_expected(
    metadata_tagging_pipeline,
    ner_simple_test_cases,
    pipeline_call_kwargs,
    raising_context,
    expected_tagging_steps_run,
):
    with raising_context:
        metadata_tagging_pipeline(ner_simple_test_cases, **pipeline_call_kwargs)
        for doc in ner_simple_test_cases:
            assert doc.metadata.get("MetadataTaggingStepsRun") == expected_tagging_steps_run
