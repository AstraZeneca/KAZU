import logging
import os.path
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Protocol

import psutil
from hydra.utils import instantiate
from omegaconf import DictConfig

from kazu.data.data import Document, PROCESSING_EXCEPTION
from kazu.steps import Step
from datetime import datetime

logger = logging.getLogger(__name__)


def load_steps_and_log_memory_usage(cfg: DictConfig) -> List[Step]:
    """
    Loads steps based on the pipeline config, and log the memory increase after loading each step.

    Note that you can instantiate the pipeline directly from the config in a way that gives the
    same results, but this is useful for monitoring/debugging high memory usage.

    :param cfg: An OmegaConf config object for the kazu :class:`.Pipeline`\\ .
        Normally created from hydra config files.
    :return: The instantiated steps from the pipeline config
    """
    steps = []
    for step in cfg.Pipeline.steps:
        prev_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        new_step = instantiate(step)
        steps.append(new_step)
        new_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        mem_increase = ((new_memory - prev_memory) / prev_memory) * 100

        logger.info(
            f"loaded {step}. Memory usage now {round(new_memory,2)} MB. Increased by {round(new_memory-prev_memory,2)}MB, {round(mem_increase,2)}%"
        )

    return steps


def calc_doc_size(doc: Document):
    return sum(len(section.text) for section in doc.sections)


def batch_metrics(docs: List[Document]):
    lengths = []
    ent_count = []
    for doc in docs:
        lengths.append(calc_doc_size(doc))
        ent_count.append(len(doc.get_entities()))
    return {
        "max_length": max(lengths),
        "mean_length": float(sum(lengths)) / float(len(docs)),
        "max_ents": max(ent_count),
        "mean_ents": float(sum(ent_count)) / float(len(ent_count)),
    }


class FailedDocsHandler(Protocol):
    """Handle failed docs."""

    def __call__(self, step_docs_map: Dict[str, List[Document]]):
        """
        :meta public:

        :param step_docs_map: a dict of step namespace and the docs that failed for it
        :return:
        """
        raise NotImplementedError()


class FailedDocsLogHandler(FailedDocsHandler):
    """Log failed docs as warnings."""

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
    """Log failed docs to a directory along with the relevant exception."""

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
        steps: List[Step],
        failure_handler: Optional[List[FailedDocsHandler]] = None,
        profile_steps_dir: Optional[str] = None,
        skip_doc_len: Optional[int] = 200000,
    ):
        """
        A basic pipeline, used to help run a series of steps

        :param steps: list of steps to run
        :param failure_handler: optional list of handlers to process failed docs
        :param profile_steps_dir: profile throughout of each step with tensorboard. path to log dir
        """
        self.skip_doc_len = skip_doc_len
        self.failure_handlers = failure_handler
        self.steps = steps
        # documents that failed to process - a dict of [<step namespace>:List[failed docs]]
        self.failed_docs: Dict[str, List[Document]] = {}
        # performance tracking
        self.init_time = datetime.now().strftime("%m_%d_%Y_%H_%M")
        logger.info(f"pipeline initialised: {self.init_time}")
        self.profile_steps_dir = profile_steps_dir
        if profile_steps_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as e:
                raise ImportError(
                    "Profiling the pipeline requires tensorboard to be installed.\n"
                    "Tensorboard is included in Kazu's dev dependencies, so this will work"
                    " with 'pip install kazu[dev]', but you could also just do 'pip install"
                    " tensorboard' if you don't want the other dev dependencies."
                ) from e

            idx_this_process = uuid.uuid1().hex
            dir_and_time = f"{profile_steps_dir}_{self.init_time}_{idx_this_process}"
            logger.info(f"profiling configured. log dir is {dir_and_time}")
            self.summary_writer: Optional[SummaryWriter] = SummaryWriter(log_dir=dir_and_time)
            self.call_count = 0
        else:
            logger.info("profiling not configured")
            self.summary_writer = None

    def prefilter_docs(self, docs: List[Document]):
        docs_to_process = []
        for doc in docs:
            doc_size = calc_doc_size(doc)
            if doc_size >= self.skip_doc_len:
                doc.metadata[
                    PROCESSING_EXCEPTION
                ] = f"document too long: {doc_size}. max:{self.skip_doc_len}"
                logger.warning(f"skipping doc: {doc.idx}: reason: too long")
            else:
                docs_to_process.append(doc)
        return docs_to_process

    def __call__(self, docs: List[Document]) -> List[Document]:
        """
        :meta public:

        run the pipeline

        :param docs: Docs to process
        :return: processed docs
        """
        docs_to_process = self.prefilter_docs(docs)
        step_times = {}
        batch_start = time.time()
        for step in self.steps:
            start = time.time()
            _processed_docs, failed_docs = step(docs_to_process)
            step_times[step.namespace()] = round(time.time() - start, 4)
            self.update_failed_docs(step, failed_docs)
        batch_time = round(time.time() - batch_start, 4)
        if self.profile_steps_dir:
            self.profile(step_times, batch_time, batch_metrics(docs))
        self.reset()

        return docs

    def profile(self, step_times: Dict, batch_time: float, batch_metrics_dict: Dict):
        if self.summary_writer is not None:
            self.summary_writer.add_scalars(
                main_tag="batch_metrics",
                tag_scalar_dict=batch_metrics_dict,
                global_step=self.call_count,
            )
            self.summary_writer.add_scalars(
                main_tag="all_steps", tag_scalar_dict=step_times, global_step=self.call_count
            )
            self.summary_writer.add_scalars(
                main_tag="batch_time",
                tag_scalar_dict={"batch": batch_time},
                global_step=self.call_count,
            )
            self.summary_writer.add_scalars(
                main_tag="memory",
                tag_scalar_dict={"MB": psutil.Process(os.getpid()).memory_info().rss / 1024**2},
                global_step=self.call_count,
            )
            for step_name, step_time in step_times.items():
                self.summary_writer.add_scalars(
                    main_tag=step_name,
                    tag_scalar_dict={"time": step_time},
                    global_step=self.call_count,
                )
            self.call_count += 1

    def update_failed_docs(self, step: Step, failed_docs: List[Document]):
        if self.failure_handlers is not None:
            self.failed_docs[step.namespace()] = failed_docs

    def reset(self):
        if self.failure_handlers is not None:
            for handler in self.failure_handlers:
                handler(self.failed_docs)
        self.failed_docs = {}
