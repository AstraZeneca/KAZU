# coding=utf-8
# Modified for distillation using Pytorch Lightning by KAZU team
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team and Huawei Noah's Ark Lab.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import List

from transformers import InputExample, DataProcessor

from kazu.modelling.distillation.data_utils import to_unicode

logger = logging.getLogger(__name__)


class SeqTagProcessor(DataProcessor):
    """Base class for data converters for sequence tagging data sets."""

    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Gets a collection of :class:`transformers.InputExample` for the train set.

        :rtype: typing.List[~transformers.InputExample]"""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """Gets a collection of :class:`transformers.InputExample` for the dev set.

        :rtype: typing.List[~transformers.InputExample]"""
        raise NotImplementedError()

    def get_aug_examples(self, data_dir) -> List[InputExample]:
        """Gets a collection of :class:`transformers.InputExample` for the dev set.

        :rtype: typing.List[~transformers.InputExample]"""
        raise NotImplementedError()


class NerProcessor(SeqTagProcessor):
    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "train_dev.tsv")), "train"
        )

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        return self._create_examples(self._read_data(os.path.join(data_dir, "devel.tsv")), "dev")

    def get_test_examples(self, data_dir: str) -> List[InputExample]:
        return self._create_examples(self._read_data(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir: str) -> List[InputExample]:
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = to_unicode(
                line[1]
            )  # TODO assert if tokenization from BERT and tokenization from pytorch mismatch
            label = to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=text, label=label))
        return examples

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        inpFilept = open(input_file)
        lines = []
        words = []
        labels = []
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

        inpFilept.close()
        logging.info("continualLineErrorCnt : %s" % (continualLineErrorCnt))
        return lines
