"""Modified for distillation using Pytorch Lightning by KAZU team.

Based off of the `DataProcessor <https://github.com/dmis-lab/biobert/blob/master/run_ner.py#L127>`_
and `NerProcessor <https://github.com/dmis-lab/biobert/blob/master/run_ner.py#L175>`_ classes in BioBERT,
also written in reference to Huawei Noah's Ark Lab `TinyBERT <https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT>`_.

Licensed under Apache 2.0

| Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team and Huawei Noah's Ark Lab.
| Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

.. raw:: html

    <details>
    <summary>Full License Notice</summary>

| Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team and Huawei Noah's Ark Lab.
| Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

.. raw:: html

    </details>
"""

import logging
import os
from collections.abc import Iterable

from transformers import InputExample, DataProcessor

from kazu.utils.utils import PathLike

logger = logging.getLogger(__name__)


# type ignore is necessary because transformers doesn't distribute type hints,
# so DataProcessor is seen as 'Any'.
class SeqTagProcessor(DataProcessor):  # type: ignore[misc]
    """Base class for data converters for sequence tagging data sets."""

    def get_train_examples(self, data_dir: str) -> list[InputExample]:
        """Gets a collection of :class:`transformers.InputExample` for the train set."""
        raise NotImplementedError

    def get_dev_examples(self, data_dir: str) -> list[InputExample]:
        """Gets a collection of :class:`transformers.InputExample` for the dev set."""
        raise NotImplementedError

    def get_aug_examples(self, data_dir: str) -> list[InputExample]:
        """Gets a collection of :class:`transformers.InputExample` for the aug set."""
        raise NotImplementedError


class NerProcessor(SeqTagProcessor):
    def get_train_examples(self, data_dir: str) -> list[InputExample]:
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "train_dev.tsv")), "train"
        )

    def get_dev_examples(self, data_dir: str) -> list[InputExample]:
        return self._create_examples(self._read_data(os.path.join(data_dir, "devel.tsv")), "dev")

    def get_test_examples(self, data_dir: str) -> list[InputExample]:
        """Gets a collection of :class:`transformers.InputExample` for the test set."""
        return self._create_examples(self._read_data(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir: str) -> list[InputExample]:
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def _create_examples(
        self, lines: Iterable[tuple[str, str]], set_type: str
    ) -> list[InputExample]:
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # TODO assert if tokenization from BERT and tokenization from pytorch mismatch
            text = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text, label=label))
        return examples

    @classmethod
    def _read_data(cls, input_file: PathLike) -> list[tuple[str, str]]:
        """Reads a BIO data."""
        with open(input_file) as inpFilept:
            lines = []
            words: list[str] = []
            labels: list[str] = []
            continualLineErrorCnt = 0
            for lineIdx, line in enumerate(inpFilept):
                contents = line.splitlines()[0]
                lineList = contents.split()
                if len(lineList) == 0:  # For blank line
                    assert len(words) == len(
                        labels
                    ), "lineIdx: %s,  len(words)(%s) != len(labels)(%s) \n %s\n%s" % (
                        lineIdx,
                        len(words),
                        len(labels),
                        " ".join(words),
                        " ".join(labels),
                    )
                    if len(words) != 0:
                        wordSent = " ".join(words)
                        labelSent = " ".join(labels)
                        lines.append((labelSent, wordSent))
                        words = []
                        labels = []
                    else:
                        continualLineErrorCnt += 1
                else:
                    words.append(lineList[0])
                    labels.append(lineList[-1])
            if len(words) != 0:
                wordSent = " ".join(words)
                labelSent = " ".join(labels)
                lines.append((labelSent, wordSent))

        logging.info("continualLineErrorCnt : %s" % (continualLineErrorCnt))
        return lines
