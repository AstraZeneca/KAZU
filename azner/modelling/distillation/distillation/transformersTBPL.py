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

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import torch
from torch import nn

from transformers import BertModel, BertPreTrainedModel

logger = logging.getLogger(__name__)

BERT_CONFIG_NAME = "bert_config.json"
TF_WEIGHTS_NAME = "model.ckpt"

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex ."
    )

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root)."""
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}
NORM = {"layer_norm": BertLayerNorm}


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        # Need to unty it when we separate the dimensions of hidden and emb
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # assert sys.version_info[0] != 2, "Python2 not supported"
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0), bias=False
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class TinyBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config, fit_size=768):
        super(TinyBertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        masked_lm_labels=None,
        next_sentence_label=None,
        labels=None,
    ):
        sequence_output, att_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask
        )
        tmp = []
        for s_id, sequence_layer in enumerate(sequence_output):
            tmp.append(self.fit_dense(sequence_layer))
        sequence_output = tmp

        return att_output, sequence_output


class TinyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels, fit_size=768):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        # self.apply(self.init_bert_weights)
        self.init_weights()

    def forward(
        self, input_ids, token_type_ids=None, attention_mask=None, labels=None, is_student=False
    ):

        # sequence_output, att_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
        #                                                       output_all_encoded_layers=True, output_att=True)
        outputHfv4 = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        # For output format of BertModel in HF v4.9.2: check https://github.com/huggingface/transformers/blob/41981a25cdd028007a7491d68935c8d93f9e8b47/src/transformers/models/bert/modeling_bert.py#L1009
        sequence_output = outputHfv4["hidden_states"]
        att_output = outputHfv4["attentions"]
        pooled_output = outputHfv4["pooler_output"]

        logits = self.classifier(torch.relu(pooled_output))

        tmp = []
        if is_student:
            for s_id, sequence_layer in enumerate(sequence_output):
                tmp.append(self.fit_dense(sequence_layer))
            sequence_output = tmp
        return logits, att_output, sequence_output


class TinyBertForSequenceTagging(BertPreTrainedModel):
    def __init__(self, config, num_labels=None, fit_size=768):
        super(TinyBertForSequenceTagging, self).__init__(config)

        if num_labels is None:
            self.num_labels = config.num_labels
        else:
            self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.seqTagger = nn.Linear(in_features=config.hidden_size, out_features=self.num_labels)
        self.classifier = self.seqTagger  # legacy TODO : delete
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        self.init_weights()

    def forward(
        self, input_ids, token_type_ids=None, attention_mask=None, labels=None, is_student=False
    ):

        outputHfv4 = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        sequence_output = outputHfv4["hidden_states"]
        # sequence_output [list of torch tensor] = (number of layers + 1) * [batch_size, sequence_length, hidden_size]
        att_output = outputHfv4["attentions"]
        logits = self.seqTagger(sequence_output[-1])

        tmp = []
        if is_student:
            for s_id, sequence_layer in enumerate(sequence_output):
                tmp.append(self.fit_dense(sequence_layer))
            sequence_output = tmp
        return logits, att_output, sequence_output
