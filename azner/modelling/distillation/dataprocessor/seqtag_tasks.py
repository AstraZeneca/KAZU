# coding=utf-8
# Modified for TODO
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

import os, sys
import csv, json
from .data_utils import to_unicode, printable_text
from .dataprocessor import DataProcessor, InputExample, InputFeatures

import logging
logger = logging.getLogger(__name__)


class SeqTagProcessor(DataProcessor):
    """Base class for data converters for sequence tagging data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_aug_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def convert_examples_to_features(self, examples, label_list, max_seq_length,
                                     tokenizer, output_mode, mode=None, batch_size=None, output_dir=None):
        """
        Loads a data file into a list of `InputBatch`s.
        output_mode : seqtag, classification, or regression
        mode : train, eval, or test
        """
    
        assert output_mode == "seqtag", "Wrong processor! Processors for sequence classification tasks must inherits SeqClsProcessor"
        
        return self.sequence_tagging_task_convert_examples_to_features(
            examples, label_list, max_seq_length, tokenizer, output_dir, mode) # mode : train, eval, or test
    
    # function convert_single_example and sequence_tagging_task_convert_examples_to_features is only for sequence tagging task
    def convert_single_example(self, ex_index, example, label_map, max_seq_length, tokenizer,mode): # For NER only
        textlist = example.text_a.split()
        labellist = example.label.split()
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m,  tok in enumerate(token):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
    
        # drop if token is longer than max_seq_length
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_id = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_id.append(label_map["O"])  # putting O instead of [CLS]
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if labels[i] == "X":
                label_id.append(label_map["O"])
            else:
                label_id.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_id.append(label_map["O"]) # putting O instead of [SEP]
    
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        
        # The mask has 1 for real tokens and 0 for padding tokens.
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)
    
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_id.append(0)
            ntokens.append("[PAD]")
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_id) == max_seq_length
    
        if ex_index < 4 : # Examples before model run
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_id: %s" % " ".join([str(x) for x in label_id]))
    
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            seq_length=seq_length
        )
        #write_tokens(ntokens,mode) # for debug
        return feature
    
    def sequence_tagging_task_convert_examples_to_features(
            self, examples, label_list, max_seq_length, tokenizer, output_dir, mode=None):
        # mode [str]: train or test
    
        label_map = {}
        for i, label in enumerate(label_list):
            label_map[label] = i
        with open(os.path.join(output_dir,'label2id.json'),'w') as w:
            json.dump(obj=label_map,fp=w)    
        
        featureSet = set()
    
        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                logger.info("Processing example %d of %d" % (ex_index, len(examples)))
            feature = self.convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer,mode)
            featureSet.add(feature)
    
        return featureSet
    



class NerProcessor(SeqTagProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train_dev.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "devel.tsv")), "dev")

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self,label_path=None):
        labelList = list()
        if label_path == None:
            labelList = ["B", "I", "O"]
            logger.info("labels.txt not found! Using basic labels of B, I, and O")
        else:
            with open(label_path, "r") as labelFp:
                for line in labelFp.readlines():
                    labelList.append(line.strip())
        return labelList

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = to_unicode(line[1]) # TODO assert if tokenization from BERT and tokenization from pytorch mismatch
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
            if len(lineList) == 0: # For blank line
                assert len(words) == len(labels), "lineIdx: %s,  len(words)(%s) != len(labels)(%s) \n %s\n%s"%(lineIdx, len(words), len(labels), " ".join(words), " ".join(labels))
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
            words = []
            labels = []

        inpFilept.close()
        logging.info("continualLineErrorCnt : %s"%(continualLineErrorCnt))
        return lines