import torch
from torch.ao.quantization import move_exported_model_to_eval
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)
from torch.export import export_for_training
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


class _Int8X86Quantizer:
    def __init__(self) -> None:
        quantization_config = get_default_x86_inductor_quantization_config(is_dynamic=True)

        quantizer = X86InductorQuantizer()
        quantizer.set_global(quantization_config)
        self.quantizer = quantizer

    @torch.inference_mode()
    def quantize(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> torch.nn.Module:
        example_inputs = tokenizer(
            "",
            max_length=max_length,
            padding=PaddingStrategy.MAX_LENGTH,
            return_tensors="pt",
        )
        example_inputs = dict(example_inputs.to(model.device))

        exported_model = export_for_training(model, args=(), kwargs=example_inputs).module()

        exported_model = prepare_pt2e(exported_model, self.quantizer)  # type: ignore[arg-type]
        exported_model(**example_inputs)

        exported_model = convert_pt2e(exported_model)
        return move_exported_model_to_eval(exported_model)  # type: ignore[no-any-return]
