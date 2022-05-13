import logging
import os.path
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from kazu.data.data import Document, PROCESSING_EXCEPTION
from kazu.steps import BaseStep
from kazu.steps.base.step import StepMetadata

from datetime import datetime

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

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir

    def __call__(self, step_docs_map: Dict[str, List[Document]]):
        for step_namespace, docs in step_docs_map.items():
            step_logging_dir = os.path.join(self.log_dir, step_namespace)
            if not os.path.exists(step_logging_dir):
                os.makedirs(step_logging_dir, exist_ok=True)

            for doc in docs:
                serialisable_doc = doc.json()
                doc_id = doc.idx
                doc_path = os.path.join(step_logging_dir, doc_id + ".json")
                doc_error_path = os.path.join(step_logging_dir, doc_id + "_error.txt")
                with open(doc_path, "w") as f:
                    f.write(serialisable_doc)
                with open(doc_error_path, "w") as f:
                    error_message = doc.metadata.get(PROCESSING_EXCEPTION)
                    if error_message is not None:
                        f.write(error_message)
                    else:
                        logger.warning(
                            f"No error message found for doc: {doc}. Cannot write exception to {doc_error_path}"
                        )


class Pipeline:
    def __init__(
        self,
        steps: List[BaseStep],
        failure_handler: Optional[List[FailedDocsHandler]] = None,
        profile_steps_dir: Optional[str] = None,
    ):
        """
        A basic pipeline, used to help run a series of steps

        :param steps: list of steps to run
        :param failure_handler: optional list of handlers to process failed docs
        :param profile_steps_dir: profile throughout of each step with tensorboard. path to log dir
        """
        self.failure_handlers = failure_handler
        self.pipeline_metadata: Dict[str, StepMetadata] = {}  # metadata about each step
        self.steps = steps
        # documents that failed to process - a dict of [<step namespace>:List[failed docs]]
        self.failed_docs: Dict[str, List[Document]] = {}
        # performance tracking
        self.init_time = datetime.now().strftime("%m_%d_%Y_%H_%M")
        logger.info(f"pipeline initialised: {self.init_time}")
        self.profile_steps_dir = profile_steps_dir
        if profile_steps_dir:
            idx_this_process = uuid.uuid1().hex
            dir_and_time = f"{idx_this_process}__{self.init_time}_{profile_steps_dir}"
            logger.info(f"profiling configured. log dir is {dir_and_time}")
            self.summary_writer = SummaryWriter(log_dir=dir_and_time)
            self.call_count = 0
        else:
            logger.info("profiling not configured")
            self.summary_writer = None

    def __call__(self, docs: List[Document]) -> List[Document]:
        """
        run the pipeline

        :param docs: Docs to process
        :return: processed docs
        """
        succeeded_docs = docs
        step_times = {}
        batch_start = time.time()
        for step in self.steps:
            start = time.time()
            succeeded_docs, failed_docs = step(succeeded_docs)
            step_times[step.namespace()] = round(time.time() - start, 4)
            self.update_failed_docs(step, failed_docs)
        batch_time = round(time.time() - batch_start, 4)
        if self.profile_steps_dir:
            self.profile(step_times, batch_time)
        self.reset()
        return succeeded_docs

    def profile(self, step_times: Dict, batch_time: float):
        if self.summary_writer is not None:
            self.summary_writer.add_scalars(
                main_tag="all_steps", tag_scalar_dict=step_times, global_step=self.call_count
            )
            self.summary_writer.add_scalars(
                main_tag="batch_time",
                tag_scalar_dict={"batch": batch_time},
                global_step=self.call_count,
            )
            for step_name, step_time in step_times.items():
                self.summary_writer.add_scalars(
                    main_tag=step_name,
                    tag_scalar_dict={"time": step_time},
                    global_step=self.call_count,
                )
            self.call_count += 1

    def update_failed_docs(self, step: BaseStep, failed_docs: List[Document]):
        if self.failure_handlers is not None:
            self.failed_docs[step.namespace()] = failed_docs

    def flush_failed_docs(self):
        if self.failure_handlers is not None:
            for handler in self.failure_handlers:
                handler(self.failed_docs)
        self.failed_docs = {}

    def reset(self):
        self.flush_failed_docs()
        self.pipeline_metadata: Dict[str, StepMetadata] = {}
