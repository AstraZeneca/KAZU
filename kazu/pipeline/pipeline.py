import logging
import os.path
from typing import List, Dict, Optional
import json
from hydra.utils import instantiate
from omegaconf import DictConfig
from fastapi.encoders import jsonable_encoder
from kazu.data.data import Document, PROCESSING_EXCEPTION
from kazu.steps import BaseStep
from kazu.steps.base.step import StepMetadata
from kazu.utils.stopwatch import Stopwatch

logger = logging.getLogger(__name__)


def load_steps(cfg: DictConfig) -> List[BaseStep]:
    """
    loads steps based on the cfg.pipeline
    """
    steps = []
    for step in cfg.pipeline.steps:
        new_step = instantiate(cfg[step])
        steps.append(new_step)

    return steps


class FailedDocsHandler:
    """
    class to somehow handle failed docs
    """

    def __call__(self, step_docs_map: Dict[str, List[Document]]):
        """

        :param step_docs_map: a dict of step namespace and the docs that failed for it
        :return:
        """
        raise NotImplementedError()


class FailedDocsLogHandler(FailedDocsHandler):
    """
    implementation that logs to warning
    """

    def __call__(self, step_docs_map: Dict[str, List[Document]]):
        for step_namespace, docs in step_docs_map.items():
            for doc in docs:
                error_message = doc.metadata.get(PROCESSING_EXCEPTION, None)
                if error_message is not None:
                    logger.warning(
                        f"processing failed for step {step_namespace}, doc: {doc.idx}, {error_message} "
                    )
                else:
                    logger.warning(
                        f"processing failed for step {step_namespace}, doc: {doc.idx}, No error mesasge found in doc metadata "
                    )


class FailedDocsFileHandler(FailedDocsHandler):
    """
    implementation logs docs to a directory, along with exception message
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir

    def __call__(self, step_docs_map: Dict[str, List[Document]]):
        for step_namespace, docs in step_docs_map.items():
            step_logging_dir = os.path.join(self.log_dir, step_namespace)
            if not os.path.exists(step_logging_dir):
                os.makedirs(step_logging_dir, exist_ok=True)

            for doc in docs:
                serialisable_doc = doc.as_serialisable()
                doc_id = doc.idx
                doc_path = os.path.join(step_logging_dir, doc_id + ".json")
                doc_error_path = os.path.join(step_logging_dir, doc_id + "_error.txt")
                with open(doc_path, "w") as f:
                    f.write(json.dumps(jsonable_encoder(serialisable_doc)))
                with open(doc_error_path, "w") as f:
                    error_message = doc.metadata.get(PROCESSING_EXCEPTION, None)
                    if error_message is not None:
                        f.write(error_message)
                    else:
                        logger.warning(
                            f"No error message found for doc: {doc}. Cannot write exception"
                        )


class Pipeline:
    def __init__(
        self, steps: List[BaseStep], failure_handler: Optional[List[FailedDocsHandler]] = None
    ):
        """
        A basic pipeline, used to help run a series of steps

        :param steps: list of steps to run
        :param failure_handler: optional list of handlers to process failed docs
        """
        self.failure_handlers = failure_handler
        self.pipeline_metadata: Dict[str, StepMetadata] = {}  # metadata about each step
        self.steps = steps
        # documents that failed to process - a dict of [<step namespace>:List[failed docs]]
        self.failed_docs: Dict[str, List[Document]] = {}
        # performance tracking
        self.stopwatch = Stopwatch()

    def check_dependencies_have_run(self, step: BaseStep):
        """
        each step can have a list of dependencies - the namespace of other steps that must have run for a step to
        be able to run. This method checks the dependencies of :param step against the pipeline_metadata, to ensure
        the step is able to run. Raises RuntimeError if this check fails

        :param step:
        :return:
        """
        for dependency in step.depends_on:
            if self.pipeline_metadata.get(dependency, False).get("has_run"):
                logger.debug(f"step: {dependency} has run")
            else:
                raise RuntimeError(
                    f"cannot run step: {step} as dependency {dependency} has not run"
                )

    def __call__(self, docs: List[Document]) -> List[Document]:
        """
        run the pipeline

        :param docs: Docs to process
        :return: processed docs
        """
        succeeded_docs = docs
        for step in self.steps:
            self.check_dependencies_have_run(step)
            self.stopwatch.start()
            succeeded_docs, failed_docs = step(succeeded_docs)
            self.stopwatch.message(
                f"{step.namespace} finished. Successful: {len(succeeded_docs)}, failed: {len(failed_docs)}"
            )
            self.update_failed_docs(step, failed_docs)
        self.reset()
        return succeeded_docs

    def update_failed_docs(self, step: BaseStep, failed_docs: List[Document]):
        self.failed_docs[step.namespace()] = failed_docs

    def flush_failed_docs(self):
        if self.failure_handlers is not None:
            for handler in self.failure_handlers:
                handler(self.failed_docs)
        self.failed_docs = {}

    def reset(self):
        self.flush_failed_docs()
        self.pipeline_metadata: Dict[str, StepMetadata] = {}
