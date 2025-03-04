# Accelerating Biomedical NER with Quantization

## Introduction

Quantization is an approach to represent model weights and/or activations using lower precision, aiming to reduce the computational costs of inference. The KAZU framework is designed for efficient and scalable document processing, without requiring a GPU. However, support for quantization on CPU is limited, as they generally lack native support for low precision data types (e.g. `bfloat16` or `int4`).

In this project, we explore the use of quantization to accelerate CPU inference for biomedical named entity recognition. Specifically, we apply 8-bit quantization to the weights and activations (`W8A8`). This enables inference speedups on CPUs supporting the `VNNI` ([Vector Neural Network Instructions](https://en.wikichip.org/wiki/x86/avx512_vnni)) instruction set extension.

## Supported hardware

The following Linux command can be used to verify if the target CPU supports `VNNI`. This should output either `avx512_vnni` or `avx_vnni` on supported systems.

```shell
lscpu | grep -o "\S*_vnni"
```

## Usage

> [!IMPORTANT]
> Quantization is currently experimental as it relies on PyTorch prototype features.

The following instructions apply to the [`TransformersModelForTokenClassificationNerStep`](https://astrazeneca.github.io/KAZU/_autosummary/kazu.steps.ner.hf_token_classification.html#kazu.steps.ner.hf_token_classification.TransformersModelForTokenClassificationNerStep).

To enable quantization, set the following environment variables. TorchInductor is required to lower the quantized model to optimized instructions.

```shell
export KAZU_ENABLE_INDUCTOR=1
export KAZU_ENABLE_QUANTIZATION=1
```

Optionally, TorchInductor [Max-Autotune](https://pytorch.org/tutorials/prototype/max_autotune_on_CPU_tutorial.html) can be enabled to automatically profile and select the best performing operation implementations.

```shell
export KAZU_ENABLE_MAX_AUTOTUNE=1
```

## Benchmarking

To benchmark inference performance, we use [`evaluate_script.py`](/kazu/training/evaluate_script.py) with the ([`multilabel_biomedBERT`](/resources/kazu_model_pack_public/multilabel_biomedBERT)) model. We use the dataset from the following guide: [training multilabel NER](https://astrazeneca.github.io/KAZU/training_multilabel_ner.html). To simulate a long workload, we use the entire test set (365 documents), whereas for a short workload, we use the first 10 documents (alphabetically).

The following benchmark results were collected on an Intel Xeon Gold 6252 CPU (single core) with PyTorch 2.6.0.

### Short workload (10 documents)

| Method                            | Mean F1 | Duration (S) | Speedup |
| :-------------------------------- | ------: | -----------: | ------: |
| Baseline                          |  0.9697 |       373.39 |    1.00 |
| Baseline (Inductor)               |  0.9697 |       357.15 |    1.05 |
| Baseline (Inductor, Max-Autotune) |  0.9697 |       358.99 |    1.04 |
| W8A8 (Inductor)                   |  0.9656 |       194.84 |    1.92 |
| W8A8 (Inductor, Max-Autotune)     |  0.9656 |       195.55 |    1.91 |

### Long workload (365 documents)

| Method                            | Mean F1 | Duration (S) | Speedup |
| :-------------------------------- | ------: | -----------: | ------: |
| Baseline                          |  0.9560 |     14797.50 |    1.00 |
| Baseline (Inductor)               |  0.9560 |     13450.01 |    1.10 |
| Baseline (Inductor, Max-Autotune) |  0.9560 |     13469.89 |    1.10 |
| W8A8 (Inductor)                   |  0.9519 |      6642.87 |    2.23 |
| W8A8 (Inductor, Max-Autotune)     |  0.9519 |      6801.01 |    2.18 |

## Conclusion

In our benchmarks, W8A8 quantization via TorchInductor achieved up to a 2&times; speedup over the baseline (W32A32) model. This incurs only a -0.4 point reduction in mean F1 score. For short workloads, the performance benefits of quantization are slightly reduced. Finally, we did not observe any additional performance benefits from using the TorchInductor Max-Autotune mode.

## Future work

- [ ] Load exported quantized models from checkpoints.
- [ ] Support mixed `int8` and `bfloat16` for speedups on newer CPUs.

## Resources

- [Tuning Guide for Deep Learning with Intel AVX-512 and Intel Deep Learning Boost on 3rd Generation Intel Xeon Scalable Processors](https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html)
- [PyTorch 2 Export Quantization with X86 Backend through Inductor](https://pytorch.org/tutorials/prototype/pt2e_quant_x86_inductor.html)
- [(prototype) PyTorch 2 Export Post Training Quantization](https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html)
- [Using Max-Autotune Compilation on CPU for Better Performance](https://pytorch.org/tutorials/prototype/max_autotune_on_CPU_tutorial.html)
