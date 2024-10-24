import logging
import math
import random
from pathlib import Path
from typing import List, Iterable, Protocol, Optional, Any, cast

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import ray
from ray import ObjectRef
from ray.util.queue import Queue, Empty, Full
from tqdm import tqdm
from pyarrow import parquet, Table

from kazu.utils.utils import PathLike, as_path
from kazu.utils.constants import HYDRA_VERSION_BASE
from kazu.pipeline import Pipeline
from kazu.data import Document, PROCESSING_EXCEPTION, Section


logger = logging.getLogger(__name__)


def chunks_with_index(lst: List[Any], n: int, index: int) -> List[Any]:
    """Yield successive n-sized chunks from lst."""
    partitions = []
    for i in range(0, len(lst), n):
        partitions.append(lst[i : i + n])

    if index + 1 > len(partitions):
        return []
    else:
        return partitions[index]


class DocumentLoader(Protocol):
    """Abstraction to load documents from a source, and converts them into
    :class:`.Document`."""

    def load(self) -> Iterable[List[Document]]:
        """Convert documents from a source into :class:`.Document`, and yield a list."""
        ...

    def batch_size(self) -> int:
        """Number of documents produced per batch."""
        ...

    def total_documents(self) -> Optional[int]:
        """Total Documents in this data source, if known."""
        ...

    def total_batches_if_available(self) -> Optional[int]:
        maybe_total = self.total_documents()
        if maybe_total is not None:
            total = int(maybe_total / self.batch_size())
        else:
            total = None
        return total


class ParquetDocumentLoader(DocumentLoader):
    def __init__(
        self,
        batch_size: int,
        source_dir: PathLike,
        randomise_processing_order: bool = True,
    ):
        """

        :param batch_size: number of documents to produce per batch
        :param source_dir: Path to parquet dataset. This should have three columns:
            id: a globally unique id for the document, ids: a dict or list of any other ids
            associated with the document, sections: an array of structs with the
            fields:
            {section:<type(str) name of section>,
            text:<type(str) the text to process>,
            subSection:<type(str) optional additional string of section information>}
        :param randomise_processing_order: should parquet files be processed in a random order?
        """
        self.randomise_processing_order = randomise_processing_order
        self._batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        self.source_dir = as_path(source_dir)
        self.files_to_process = self._get_file_list()
        self.logger.info(f"{len(self.files_to_process)} file to do this batch run")

    def batch_size(self) -> int:
        return self._batch_size

    def _list_files_in_dir(self, dir: Path) -> List[Path]:
        paths = []
        for path in dir.iterdir():
            # ignore any non-parquet files
            if path.suffix == ".parquet":
                paths.append(path)
        return sorted(paths)

    def _get_file_list(self) -> List[Path]:
        self.logger.info(f"selecting all files from {self.source_dir}")
        todo_as_paths = self._list_files_in_dir(self.source_dir)
        if self.randomise_processing_order:
            random.shuffle(todo_as_paths)
        else:
            todo_as_paths.sort()
        return todo_as_paths

    def _table_slice_to_docs(self, table: Table) -> List[Document]:
        docs = []

        for as_dict in table.select(["id", "sections"]).to_pylist():
            sections = as_dict["sections"]
            idx = as_dict["id"]
            kazu_sections = []
            for section in sections:
                kazu_sections.append(
                    Section(
                        name=section["section"],
                        text=section["text"],
                        metadata={"subSection": section.get("subSection")},
                    )
                )
            docs.append(Document(idx=idx, sections=kazu_sections))
        return docs

    def load(self) -> Iterable[List[Document]]:
        for target_file in self.files_to_process:
            for docs in self._subpartition_parquet(target_file):
                yield docs

    def total_documents(self) -> int:
        table = parquet.read_table(self.source_dir, columns=["id"])
        return cast(int, table.shape[0])

    def _subpartition_parquet(self, file_path: Path) -> Iterable[List[Document]]:
        table = parquet.read_table(file_path)
        if table.shape[0] == 0:
            self.logger.debug(
                f"no rows detected/required in file {file_path}. Nothing to partition"
            )
        else:
            partition_count = math.ceil(table.shape[0] / self.batch_size())
            self.logger.info(
                f"dataframe will yield {partition_count} partitions ({table.shape[0]} rows)"
            )
            offset_index = 0
            while True:
                slice = table.slice(offset_index, self.batch_size())
                if slice.shape[0] == 0:
                    self.logger.info(f"no more slices for {file_path}")
                    break
                else:
                    docs = self._table_slice_to_docs(slice)
                    yield docs
                    offset_index = offset_index + self.batch_size()


class RayPipelineQueueWorker:
    """Reads Documents from a queue, processes with a pipeline and writes to another
    queue."""

    def __init__(self, cfg: DictConfig):
        self.pipeline: Pipeline = instantiate(cfg.pipeline, _convert_="all")
        self.logger = logging.getLogger(__name__)
        self.in_queue: Optional[Queue] = None
        self.out_queue: Optional[Queue] = None

    def run(self) -> None:
        while True:
            docs: list[Document] = await_get(queue=self.in_queue, timeout=None)  # type: ignore[arg-type]
            self.pipeline(docs)
            await_put(self.out_queue, docs, timeout=None)  # type: ignore[arg-type]

    def set_queues(self, in_queue: Queue, out_queue: Queue) -> None:
        self.in_queue = in_queue
        self.out_queue = out_queue


def await_put(queue: Queue, items: Any, timeout: Optional[float] = 300.0) -> None:
    """Put items into a queue.

    Retries until successful
    """
    attempt = 1
    while True:
        try:
            queue.put(items, block=True, timeout=timeout)
            return
        except Full:
            logger.info(f"cannot put into queue: Full. Attempt: {attempt}")
            attempt += 1


def await_get(queue: Queue, timeout: Optional[float] = 300.0) -> List[Document]:
    """Gets items from a queue."""
    try:
        items = cast(list[Document], queue.get(block=True, timeout=timeout))
        return items
    except Empty:
        return []


class RayBatchRunner:
    def __init__(
        self,
        cfg: DictConfig,
    ):
        self.cfg = cfg
        self.worker_count = cfg.worker_count
        ray.init(num_cpus=self.worker_count)
        self.loader: ParquetDocumentLoader = instantiate(cfg.ParquetDocumentLoader)
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(exist_ok=True)
        self.failed_dir = Path(cfg.failed_dir)
        self.failed_dir.mkdir(exist_ok=True)

        self.in_queue: Queue = Queue(maxsize=self.worker_count * 2)
        self.out_queue: Queue = Queue(maxsize=100)
        self.workers: List[ObjectRef[None]] = self.instantiate_workers()

    def instantiate_workers(self) -> List[ObjectRef]:  # type: ignore[type-arg]
        worker_refs = []
        for i in range(self.worker_count):
            worker_refs.append(self._instantiate_worker())
        return worker_refs

    def _instantiate_worker(self) -> ObjectRef:  # type: ignore[type-arg]
        PipelineQueueWorkerActor = ray.remote(RayPipelineQueueWorker)
        PipelineQueueWorkerActor.options(num_cpus=1)  # type: ignore[attr-defined]

        worker: ObjectRef = PipelineQueueWorkerActor.remote(self.cfg)  # type: ignore[type-arg]
        worker.set_queues.remote(self.in_queue, self.out_queue)  # type: ignore[attr-defined]
        task: ObjectRef = worker.run.remote()  # type: ignore[type-arg,attr-defined]
        print("worker started")
        return task

    def check_doc_not_already_processed(self, docs: List[Document]) -> List[Document]:
        unprocessed = []
        for doc in docs:
            out_path = self.out_dir.joinpath(f"{doc.idx}.json")
            failed_path = self.failed_dir.joinpath(f"{doc.idx}.json")
            if out_path.exists() or failed_path.exists():
                print(f"skipping {doc.idx}")
                continue
            unprocessed.append(doc)
        return unprocessed

    def start(self) -> Iterable[List[Document]]:
        docs_wanted = 0
        responses = 0
        log_count = 0
        for i, docs in enumerate(
            tqdm(self.loader.load(), smoothing=0.1, total=self.loader.total_batches_if_available())
        ):
            docs_wanted += len(docs)
            unprocessed = self.check_doc_not_already_processed(docs)
            responses += len(docs) - len(unprocessed)
            await_put(queue=self.in_queue, items=unprocessed, timeout=30.0)
            if not self.out_queue.empty():
                result = await_get(queue=self.out_queue, timeout=30.0)
                responses += len(result)
                yield result
            else:
                if log_count % 10 == 0:
                    logger.info("awaiting results: %s / %s", docs_wanted, responses)
                log_count += 1

        while docs_wanted != responses:
            logger.info("awaiting final batches: %s / %s", docs_wanted, responses)
            result = await_get(queue=self.out_queue, timeout=None)
            responses += len(result)
            yield result


@hydra.main(version_base=HYDRA_VERSION_BASE, config_path="conf", config_name="config")
def run_pipe(cfg: DictConfig) -> None:
    job = RayBatchRunner(cfg.annotate_with_llm)
    success_count = 0
    fail_count = 0
    for docs in tqdm(job.start(), total=job.loader.total_batches_if_available()):
        for doc in docs:
            out_path = job.out_dir.joinpath(f"{doc.idx}.json")
            failed_path = job.failed_dir.joinpath(f"{doc.idx}.json")
            if PROCESSING_EXCEPTION in doc.metadata:
                if "authentication" in doc.metadata[PROCESSING_EXCEPTION].lower():
                    raise RuntimeError(f"Google auth error: {doc.metadata[PROCESSING_EXCEPTION]}")
                else:
                    print(doc.metadata)
                    fail_count += 1
                    with failed_path.open(mode="w") as f:
                        f.write(doc.to_json())

            else:
                print(f"writing to {out_path}")
                with out_path.open(mode="w") as f:
                    f.write(doc.to_json())
                success_count += 1
        try:
            prop = success_count / (success_count + fail_count)
        except ZeroDivisionError:
            prop = 0
        print(f"success: {success_count}, fail: {fail_count}, {prop}% success")


if __name__ == "__main__":
    run_pipe()
