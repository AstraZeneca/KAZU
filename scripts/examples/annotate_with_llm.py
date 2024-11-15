import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, cast

import hydra
import ray
from hydra.utils import instantiate
from omegaconf import DictConfig
from ray import ObjectRef
from ray.util.queue import Empty, Full, Queue
from tqdm import tqdm

from kazu.data import PROCESSING_EXCEPTION, Document
from kazu.pipeline import Pipeline
from kazu.utils.constants import HYDRA_VERSION_BASE

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
        self.batch_size = cfg.batch_size
        self.source_dir = Path(cfg.source_dir)
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
        PipelineQueueWorkerActor.options(num_cpus=1)  # type: ignore[attr-defined,unused-ignore]

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
        for _, docs in enumerate(
            tqdm(self.load_documents_in_batches(), smoothing=0.1, total=len(self.get_docs_paths))
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

    @property
    def get_docs_paths(self) -> List[Path]:
        return list(self.source_dir.glob("*.json"))

    def load_documents_in_batches(self) -> Iterable[List[Document]]:
        for i in range(0, len(self.get_docs_paths), self.batch_size):
            paths_batch = self.get_docs_paths[i : i + self.batch_size]
            doc_batch = []
            for doc_path in paths_batch:
                with doc_path.open(mode="r") as f:
                    doc_batch.append(Document.from_json(f.read()))
            yield doc_batch


@hydra.main(version_base=HYDRA_VERSION_BASE, config_path="conf", config_name="config")
def run_pipe(cfg: DictConfig) -> None:
    job = RayBatchRunner(cfg.annotate_with_llm)
    success_count = 0
    fail_count = 0
    for docs in tqdm(job.start(), total=len(job.get_docs_paths)):
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
