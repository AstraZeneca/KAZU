from typing import Optional, Union, Tuple

import torch
from torch import nn
from transformers import (
    DistilBertForTokenClassification,
    BertForTokenClassification,
    PretrainedConfig,
)
from transformers.modeling_outputs import TokenClassifierOutput, BaseModelOutput
from transformers.utils import check_min_version

check_min_version("4.9.0")  # transformers version check


def multi_label_forward(
    model: nn.Module,
    outputs: BaseModelOutput,
    return_dict: bool,
    device: torch.device,
    ignore_index: int,
    labels: Optional[torch.Tensor] = None,
) -> Union[TokenClassifierOutput, Tuple[torch.Tensor, ...]]:
    sequence_output = outputs[0]
    sequence_output = model.dropout(sequence_output)
    logits = model.classifier(sequence_output)
    loss = None
    if labels is not None:
        loss_fct = nn.BCEWithLogitsLoss()
        ignore_index_loc = labels.mean(-1).squeeze().int()

        flat_outputs = logits.squeeze()[ignore_index_loc != ignore_index]
        flat_labels = labels.squeeze()[ignore_index_loc != ignore_index]

        flat_outputs = flat_outputs.to(device)
        flat_labels = flat_labels.to(device)

        loss = loss_fct(flat_outputs, flat_labels)
    if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return TokenClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


class DistilBertForMultiLabelTokenClassification(DistilBertForTokenClassification):  # type: ignore[misc] # ignored due to no transformer stubs

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.ignore_index = -100

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TokenClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""Labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length,
        self.num_labels)`, `optional`):

        Labels for computing the token classification loss. Indices should be in ``[0, 1]``.
        Each token has a vector of self.num_labels float values.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return multi_label_forward(
            model=self,
            outputs=outputs,
            return_dict=return_dict,
            device=self.device,
            ignore_index=self.ignore_index,
            labels=labels,
        )


class BertForMultiLabelTokenClassification(BertForTokenClassification):  # type: ignore[misc] # ignored due to no transformer stubs

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.ignore_index = -100

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TokenClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""Labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`,
        *optional*):

        Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return multi_label_forward(
            model=self,
            outputs=outputs,
            return_dict=return_dict,
            device=self.device,
            ignore_index=self.ignore_index,
            labels=labels,
        )
