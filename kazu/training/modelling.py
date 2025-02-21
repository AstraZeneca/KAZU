from typing import Any, Optional, Tuple, Union

import torch
from torch import nn
from transformers import (
    AutoConfig,
    DistilBertForTokenClassification,
    BertForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
    DebertaV2ForTokenClassification,
)
from transformers.modeling_outputs import TokenClassifierOutput, BaseModelOutput
from transformers.utils import check_min_version

check_min_version("4.9.0")  # transformers version check


def multi_label_forward(
    model: PreTrainedModel,
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


class DebertaForMultiLabelTokenClassification(DebertaV2ForTokenClassification):  # type:ignore[misc]
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.ignore_index = -100

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:  # type:ignore[type-arg]
        """

        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param position_ids:
        :param inputs_embeds:
        :param labels:
        :param output_attentions:
        :param output_hidden_states:
        :param return_dict:
        :return:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
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
        """

        :param input_ids:
        :param attention_mask:
        :param head_mask:
        :param inputs_embeds:
        :param labels:
        :param output_attentions:
        :param output_hidden_states:
        :param return_dict:
        :return:
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
        """

        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param position_ids:
        :param head_mask:
        :param inputs_embeds:
        :param labels:
        :param output_attentions:
        :param output_hidden_states:
        :param return_dict:
        :return:
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


class AutoModelForMultiLabelTokenClassification:
    _model_mapping: dict[str, type[PreTrainedModel]] = {
        "deberta": DebertaForMultiLabelTokenClassification,
        "distilbert": DistilBertForMultiLabelTokenClassification,
        "bert": BertForMultiLabelTokenClassification,
    }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> PreTrainedModel:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model_class = cls._model_mapping.get(config.model_type)

        if model_class is None:
            raise ValueError(
                f"Model type `{config.model_type}` is not supported for multi-label token classification."
            )

        return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
