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

import sys
import csv

import torch
from torch.utils.data import TensorDataset, DataLoader

from azner.modelling.distillation.dataprocessor.data_utils import to_unicode

import logging
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(to_unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

def get_tensor_data(output_mode, features):
    if output_mode == "seqtag":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    elif output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


def get_data_loader(args, data_type, processor, label_list, tokenizer, output_mode):
    """
    Train and validate data preparation.
    TODO : make multi-processing-enabled Custom Dataset class with 
        {get_train_examples, get_aug_examples, get_dev_examples}, convert_examples_to_features and get_tensor_data functions. 

    :param data_type: [description]
    :type data_type: [type]
    """
    if data_type == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif data_type == "aug_train":
        examples = processor.get_aug_examples(args.data_dir)
    elif data_type == "development":
        examples = processor.get_dev_examples(args.data_dir)
    len_examples = len(examples)

    data_mode = "train" if data_type in ["train", "aug_train"] else "eval"
    shuffle = True if data_type in ["train", "aug_train"] else False

    features = processor.convert_examples_to_features(
        examples, label_list, args.max_seq_length, tokenizer, output_mode, 
        mode=data_mode, batch_size=args.train_batch_size, output_dir=args.output_dir
    ) # WJ: TODO: bottle-neck
    data, labels = get_tensor_data(output_mode, features)
    dataloader = DataLoader(data, shuffle=shuffle, batch_size=args.eval_batch_size, num_workers=args.num_workers) # by default, DistributedSampler will be used by PL
    return dataloader, len_examples