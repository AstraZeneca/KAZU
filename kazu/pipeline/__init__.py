from datetime import datetime
import logging
import os
import time
import uuid
from typing import Optional, Protocol
from collections.abc import Iterable, Collection

import psutil
from hydra.utils import instantiate
from omegaconf import DictConfig

from kazu.data import Document, PROCESSING_EXCEPTION
from kazu.steps import Step
from kazu.utils.utils import PathLike, as_path

logger = logging.getLogger(__name__)


def load_steps_and_log_memory_usage(cfg: DictConfig) -> list[Step]:
    """Loads steps based on the pipeline config, and log the memory increase after
    loading each step.

    Note that you can instantiate the pipeline directly from the config in a way that gives the
    same results, but this is useful for monitoring/debugging high memory usage.

    :param cfg: An OmegaConf config object for the kazu :class:`.Pipeline`\\ .
        Normally created from hydra config files.
    :return: The instantiated steps from the pipeline config
    """
    steps = []
    for step_cfg in cfg.Pipeline.steps:
        prev_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        new_step = instantiate(step_cfg, _convert_="all")
        steps.append(new_step)
        new_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        mem_increase = ((new_memory - prev_memory) / prev_memory) * 100

        logger.info(
            f"loaded {new_step.namespace()}. Memory usage now {round(new_memory,2)} MB. Increased by {round(new_memory-prev_memory,2)}MB, {round(mem_increase,2)}%"
        )

    return steps


def calc_doc_size(doc: Document) -> int:
    return sum(len(section.text) for section in doc.sections)


def batch_metrics(docs: list[Document]) -> dict[str, float]:
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

    def __call__(self, step_docs_map: dict[str, list[Document]]) -> None:
        """
        :param step_docs_map: a dict of step namespace and the docs that failed for it
        :return:
        """
        raise NotImplementedError


class FailedDocsLogHandler(FailedDocsHandler):
    """Log failed docs as warnings."""

    def __call__(self, step_docs_map: dict[str, list[Document]]) -> None:
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

    def __init__(self, log_dir: PathLike):
        self.log_dir = as_path(log_dir)

    def __call__(self, step_docs_map: dict[str, list[Document]]) -> None:
        for step_namespace, docs in step_docs_map.items():
            step_logging_dir = self.log_dir.joinpath(step_namespace)
            step_logging_dir.mkdir(parents=True, exist_ok=True)

            for doc in docs:
                serialisable_doc = doc.to_json()
                doc_id = doc.idx
                doc_path = step_logging_dir.joinpath(doc_id + ".json")
                doc_error_path = step_logging_dir.joinpath(doc_id + "_error.txt")
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


class PipelineValueError(ValueError):
    """An Exception that gets raised if arguments passed to a Pipeline are
    inappropriate.

    This is particularly useful to allow handling just these errors - e.g. in Kazu's web API,
    where we can then highlight that there is a problem with the data passed to the API, rather
    than an internal error to Kazu.
    """

    pass


class Pipeline:
    def __init__(
        self,
        steps: list[Step],
        failure_handler: Optional[list[FailedDocsHandler]] = None,
        profile_steps_dir: Optional[str] = None,
        skip_doc_len: Optional[int] = 200000,
        step_groups: Optional[dict[str, Collection[str]]] = None,
    ):
        """A basic pipeline, used to help run a series of steps.

        :param steps: list of steps to run
        :param failure_handler: optional list of handlers to process failed docs
        :param profile_steps_dir: profile throughout of each step with tensorboard. path to log dir
        :param skip_doc_len: a maximum length for documents (in characters), above which they will
            be skipped. Extremely long inputs can be symptomatic of very strange text which can
            result in errors and excessive memory usage.
        :param step_groups: groups of steps to make available for running together as a
            convenience. The keys are names of groups to create, and values are the namespaces of
            steps. Order of running steps is still taken from the steps argument, not the order
            within each group. To customize step running order, you can instead use the
            ``step_namespaces`` parameter of :meth:`__call__`\\ .
        """
        self.skip_doc_len = skip_doc_len
        self.failure_handlers = failure_handler
        self.steps = steps
        self._namespace_to_step = {step.namespace(): step for step in steps}
        # documents that failed to process - a dict of [<step namespace>:list[failed docs]]
        self.failed_docs: dict[str, list[Document]] = {}
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

        self.step_groups: Optional[dict[str, list[Step]]]
        if step_groups is None:
            self.step_groups = None
        else:
            self.step_groups = {group_name: [] for group_name in step_groups}
            for step_name, step in self._namespace_to_step.items():
                for group_name, group in step_groups.items():
                    if step_name in group:
                        self.step_groups[group_name].append(step)

    def prefilter_docs(self, docs: list[Document]) -> list[Document]:
        if self.skip_doc_len is None:
            # we don't filter out any docs due to length
            return docs
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

    def __call__(
        self,
        docs: list[Document],
        step_namespaces: Optional[Iterable[str]] = None,
        step_group: Optional[str] = None,
    ) -> list[Document]:
        """Run the pipeline.

        :param docs: Docs to process
        :param step_namespaces: The namespaces of the steps to use in processing. Default behaviour
            is to use all steps on the pipeline in the order given when creating the pipeline.
            This parameter gives the flexibility to sometimes only run some of the steps.
        :param step_group: One of the pipeline's configured step_groups to run. This is just a
            convenience over needing to specify common groups of step_namespaces. Note that passing
            both step_group and step_namespaces is incohorent and a :exc:`ValueError` will be
            raised.
        :return: processed docs
        """
        docs_to_process = self.prefilter_docs(docs)
        step_times = {}
        batch_start = time.time()

        steps_to_run: Iterable[Step]
        if step_namespaces is not None:
            if step_group is not None:
                raise PipelineValueError(
                    "Passing both step_namespaces and step_group to Pipeline.__call__ is incomptable."
                    " Only one may be passed in a single call."
                )

            steps_to_run = []
            nonexistent_steps = []
            for namespace in step_namespaces:
                maybe_step = self._namespace_to_step.get(namespace)
                if maybe_step is None:
                    nonexistent_steps.append(namespace)
                else:
                    steps_to_run.append(maybe_step)

            if len(nonexistent_steps) >= 1:
                raise PipelineValueError(
                    "The following steps do not exist in the pipeline:\n%s"
                    "The valid steps for this pipeline are:\n%s"
                    % (nonexistent_steps, list(self._namespace_to_step))
                )

        elif step_group is not None:
            if self.step_groups is None:
                raise PipelineValueError(
                    "This pipeline does not have any step groups configured, so cannot run the"
                    " requested step_group %s" % step_group
                )

            # we can't assign directly to steps_to_run, because the type is Optional[list[Step]]
            # rather than the Iterable[Step] declared above (and required below).
            steps_or_none = self.step_groups.get(step_group)
            if steps_or_none is None:
                raise PipelineValueError(
                    "%s is not a valid step_group for this pipeline. Available step_groups:\n%s"
                    % (step_group, self.step_groups)
                )
            else:
                steps_to_run = steps_or_none

        else:
            steps_to_run = self.steps

        for step in steps_to_run:
            start = time.time()
            _processed_docs, failed_docs = step(docs_to_process)
            step_times[step.namespace()] = round(time.time() - start, 4)
            self.update_failed_docs(step, failed_docs)
        batch_time = round(time.time() - batch_start, 4)
        if self.profile_steps_dir:
            self.profile(step_times, batch_time, batch_metrics(docs))
        self.reset()

        return docs

    def profile(self, step_times: dict, batch_time: float, batch_metrics_dict: dict) -> None:
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

    def update_failed_docs(self, step: Step, failed_docs: list[Document]) -> None:
        if self.failure_handlers is not None:
            self.failed_docs[step.namespace()] = failed_docs

    def reset(self) -> None:
        if self.failure_handlers is not None:
            for handler in self.failure_handlers:
                handler(self.failed_docs)
        self.failed_docs = {}
