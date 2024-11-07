from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    #: passed to .from_pretrained in transformers
    hf_name: str
    #: past to test kazu documents
    test_path: Path
    #: past to train kazu documents
    train_path: Path
    #: directory to save training data cache
    training_data_cache_dir: Path
    #: directory to save test data cache
    test_data_cache_dir: Path
    #: directory to save output
    working_dir: Path
    #: max sequence length per training instance
    max_length: int
    #: use cache for training data (otherwise tensors will be regenerated)
    use_cache: bool
    #: max number of documents to use. None for all
    max_docs: Optional[int]
    #: stride for splitting documents into training instances (see HF tokenizers)
    stride: int
    #: batch size
    batch_size: int
    #: learning rate
    lr: float
    #: evaluate at every n step intervals
    evaluate_at_step_interval: int
    #: number of epochs to train for
    num_epochs: int
    #: warmup proportion for lr scheduler
    lr_scheduler_warmup_prop: float
    #: whether to test on a small dummy dataset (for debugging)
    test_overfit: bool
    #: device to train on
    device: str
    #: number of workers for dataloader
    workers: int
    #: architecture to use. Currently supports bert, deberta, distilbert
    architecture: str = "bert"
    #: fraction of epoch to complete before evaluations begin
    epoch_completion_fraction_before_evals: float = 0.75


@dataclass
class PredictionConfig:
    #: path to the trained model
    path: Path
    #: batch size
    batch_size: int
    #: stride for splitting documents into training instances (see HF tokenizers)
    stride: int
    #: max sequence length per training instance
    max_sequence_length: int
    #: device to train on
    device: str
    #: architecture to use. Currently supports bert, deberta, distilbert
    architecture: str = "bert"
