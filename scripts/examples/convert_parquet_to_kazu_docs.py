import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, cast

import hydra
import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig
from pyarrow import Table, parquet

from kazu.data import Document, Section
from kazu.utils.constants import HYDRA_VERSION_BASE
from kazu.utils.utils import PathLike, as_path


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


def merge_documents(docs: set[Document]) -> Document:
    # Assuming these docs are near duplicates, we can keep the one with the most sections for now
    return max(docs, key=lambda x: len(x.sections))


@hydra.main(version_base=HYDRA_VERSION_BASE, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    document_loader: ParquetDocumentLoader = instantiate(
        cfg.convert_parquet_to_kazu_docs.ParquetDocumentLoader
    )
    output_dir = Path(cfg.convert_parquet_to_kazu_docs.out_dir)
    output_dir.mkdir(exist_ok=True)

    docs_by_idx = defaultdict(set)
    for docs in tqdm.tqdm(
        document_loader.load(), total=document_loader.total_batches_if_available()
    ):
        for doc in docs:
            docs_by_idx[doc.idx].add(doc)

    loaded_docs = []
    for idx, duplicate_docs in docs_by_idx.items():
        if len(duplicate_docs) == 1:
            loaded_docs.append(duplicate_docs.pop())
        else:
            print(
                f"Found {len(duplicate_docs)} duplicates for {idx}. Need to merge them to avoid duplicates."
            )
            merged_doc = merge_documents(duplicate_docs)
            loaded_docs.append(merged_doc)

    print(f"After dropping duplicates left with {len(loaded_docs)} documents to save.")
    for doc in loaded_docs:
        file_path = output_dir / f"{doc.idx}.json"
        with file_path.open("w") as f:
            f.write(doc.to_json())


if __name__ == "__main__":
    main()
