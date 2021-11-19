import logging
from typing import List, Dict

from hydra.utils import instantiate
from omegaconf import DictConfig

from azner.data.data import Document
from azner.steps import BaseStep
from azner.steps.base.step import StepMetadata
from azner.utils.stopwatch import Stopwatch

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


class Pipeline:
    def __init__(self, steps: List[BaseStep]):
        """
        A basic pipeline, used to help run a series of steps
        :param steps: list of steps to run
        """
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
            self.update_metadata(step, StepMetadata(has_run=True))
            self.update_failed_docs(step, failed_docs)

        return docs

    def update_metadata(self, step: BaseStep, metadata: StepMetadata):
        self.pipeline_metadata[step.namespace()] = metadata

    def update_failed_docs(self, step: BaseStep, failed_docs: List[Document]):
        self.failed_docs[step.namespace()] = failed_docs
