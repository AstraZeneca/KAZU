import os
from multiprocessing import freeze_support
from pathlib import Path
from typing import Iterable

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
import hydra

from kazu.training.config import TrainingConfig
from kazu.training.train_multilabel_ner import Trainer, KazuNerMultiLabelTrainingDataset
from kazu.utils.constants import HYDRA_VERSION_BASE
from kazu.data import Document, Section, Entity, ENTITY_OUTSIDE_SYMBOL
from kazu.utils.utils import PathLike


def doc_yielder(path: PathLike) -> Iterable[Document]:
    for file in Path(path).iterdir():
        with file.open(mode="r") as f:
            try:
                yield Document.from_json(f.read())
            except Exception as e:
                print(f"failed to read: {file}, {e}")


def test_doc_yielder() -> Iterable[Document]:
    section = Section(text="abracodabravir detameth targets BEHATHT.", name="test1")
    section.entities.append(
        Entity.load_contiguous_entity(
            start=0, end=23, match="abracodabravir detameth", entity_class="drug", namespace="test"
        )
    )
    section.entities.append(
        Entity.load_contiguous_entity(
            start=15, end=23, match="detameth", entity_class="salt", namespace="test"
        )
    )
    section.entities.append(
        Entity.load_contiguous_entity(
            start=32, end=39, match="BEHATHT", entity_class="gene", namespace="test"
        )
    )
    doc = Document(sections=[section])
    yield doc


if os.getenv("KAZU_MODEL_PACK"):
    config_path = str(Path(str(os.getenv("KAZU_MODEL_PACK"))).joinpath("conf"))
else:
    raise ValueError("KAZU_MODEL_PACK not set")


@hydra.main(version_base=HYDRA_VERSION_BASE, config_path=config_path, config_name="config")
def run(cfg: DictConfig) -> None:
    training_config: TrainingConfig = instantiate(cfg.training)
    print(os.environ)
    hf_name = training_config.hf_name
    cache_dir = training_config.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(hf_name, cache_dir=cache_dir)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    tb_path = output_dir.joinpath("tensorboard")
    tb_path.mkdir(exist_ok=True)
    print(f'run tensorboard --logdir "{str(tb_path.absolute())}" to see training progress')
    training_data_cache_dir = Path(training_config.training_data_cache_dir)
    training_data_cache_dir.mkdir(exist_ok=True)
    eval_data_cache_dir = Path(training_config.eval_data_cache_dir)
    eval_data_cache_dir.mkdir(exist_ok=True)
    models_dir = output_dir.joinpath("models")
    models_dir.mkdir(exist_ok=True)
    if training_config.test_overfit:
        print("running in test mode")
        label_list = ["drug", "gene", "salt", ENTITY_OUTSIDE_SYMBOL]
        train_doc_iter = test_doc_yielder()
        test_doc_iter = test_doc_yielder()
    else:
        label_list = get_label_list(training_config.eval_path)
        train_doc_iter = doc_yielder(training_config.train_path)
        test_doc_iter = doc_yielder(training_config.eval_path)
    print(f"labels are :{label_list}")

    train_ds = KazuNerMultiLabelTrainingDataset(
        docs_iter=train_doc_iter,
        model_tokenizer=tokenizer,
        labels=label_list,
        tmp_dir=training_data_cache_dir,
        max_length=training_config.max_length,
        use_cache=training_config.use_cache,
        max_docs=training_config.max_docs,
        stride=training_config.stride,
    )
    eval_ds = KazuNerMultiLabelTrainingDataset(
        docs_iter=test_doc_iter,
        model_tokenizer=tokenizer,
        labels=label_list,
        tmp_dir=eval_data_cache_dir,
        max_length=cfg.training.max_length,
        use_cache=cfg.training.use_cache,
        max_docs=None,
        stride=cfg.training.stride,
        keep_doc_reference=True,
    )

    trainer = Trainer(
        training_config=training_config,
        pretrained_model_name_or_path=hf_name,
        label_list=label_list,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        working_dir=output_dir,
        summary_writer=SummaryWriter(log_dir=str(tb_path.absolute())),
    )
    trainer.train_model()


def get_label_list(path: PathLike) -> list[str]:
    label_list = set()
    for doc in doc_yielder(path):
        # TODO clean data as LLM is a bit weird!
        for ent in doc.get_entities():
            maybe_number = ent.entity_class[0:2].strip()
            if maybe_number.isdigit():
                label_list.add(maybe_number)
            else:
                maybe_number = maybe_number[0]
                label_list.add(maybe_number)

    label_list.add(ENTITY_OUTSIDE_SYMBOL)
    label_list.remove("10")
    return list(label_list)


if __name__ == "__main__":
    freeze_support()
    run()
