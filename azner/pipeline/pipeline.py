import logging
from typing import List, Dict, Type

from hydra.utils import instantiate
from omegaconf import DictConfig

from azner.steps import BaseStep
from azner.data.data import Document
from azner.steps.base.step import StepMetadata
from azner.utils.stopwatch import Stopwatch

logger = logging.getLogger(__name__)


def load_steps(cfg: DictConfig) -> List[BaseStep]:
    """
    loads steps based on the cfg.pipeline
    This function looks for subclasses of BaseStep, and initialises them via hydra instantiate.
    The hydra config is passed to the step constructor based on the class name. Therefore, to
    instantiate instances of BaseStep via this method, the following must be true:

    1) the module containing the Step to be instantiated must be loaded before this method is called
    2) a hydra configuration directory must be available, corresponding to the name of the Step to be
    instantiated
    3) this hydra yaml must follow the following pattern:

    _target_: # fully qualified name of class to be instantiated
    <list of kwargs> :  #for the __init__ method of the class

     e.g.
     _target_: azner.steps.ner.hf_token_classification.TransformersModelForTokenClassificationNerStep
    path: ~
    depends_on: ~

    :param cfg: Hydra DictConfig
    :return: List of Step instances
    """

    def get_all_subclasses(cls: Type) -> List[Type]:
        """
        recursively find all subclasses for a given type
        :param cls: query type
        :return: list of subclasses
        """
        all_subclasses = []
        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(get_all_subclasses(subclass))

        return all_subclasses

    base_step_subclasses = get_all_subclasses(BaseStep)
    base_step_subclasses_names = [x.__name__ for x in base_step_subclasses]
    steps = []
    for step in cfg.pipeline.steps:
        if step in base_step_subclasses_names:
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
