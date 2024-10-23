from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    hf_name: str
    cache_dir: str
    eval_path: Path
    train_path: Path
    training_data_cache_dir: Path
    eval_data_cache_dir: Path
    working_dir: Path
    max_length: int
    use_cache: bool
    max_docs: Optional[int]
    stride: int
    batch_size: int
    lr: float
    evaluate_at_step_interval: int
    num_epochs: int
    lr_scheduler_warmup_prop: float
    test_overfit: bool
    device: str
    workers: int
    architecture: str = "bert"
