# coding=utf-8
# Modified for distillation using Pytorch Lightning by KAZU team
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team., and KAZU Team
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
"""PyTorch BERT model."""
from torch import nn

from transformers import BertModel, BertPreTrainedModel


class TinyBertForSequenceTagging(BertPreTrainedModel):
    def __init__(self, config, num_labels=None, fit_size=768):
        super(TinyBertForSequenceTagging, self).__init__(config)

        if num_labels is None:
            self.num_labels = config.num_labels
        else:
            self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.seq_tagger = nn.Linear(in_features=config.hidden_size, out_features=self.num_labels)
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        self.init_weights()

    def forward(
        self, input_ids, token_type_ids=None, attention_mask=None, labels=None, is_student=False
    ):

        output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        sequence_output = output["hidden_states"]
        # sequence_output [list of torch tensor] = (number of layers + 1) * [batch_size, sequence_length, hidden_size]
        att_output = output["attentions"]
        logits = self.seq_tagger(sequence_output[-1])

        tmp = []
        if is_student:
            for s_id, sequence_layer in enumerate(sequence_output):
                tmp.append(self.fit_dense(sequence_layer))
            sequence_output = tmp
        return logits, att_output, sequence_output
