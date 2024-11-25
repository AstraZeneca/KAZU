import json
from pathlib import Path
from typing import Iterable, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig

from kazu.annotation.label_studio import (
    LabelStudioAnnotationView,
    LabelStudioManager,
)
from kazu.data import ENTITY_OUTSIDE_SYMBOL, Document, Entity, Section
from kazu.training.train_multilabel_ner import (
    LSManagerViewWrapper,
)
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


def get_label_list(path: PathLike) -> list[str]:
    label_list = set()
    for doc in doc_yielder(path):
        for entity in doc.get_entities():
            label_list.add(entity.entity_class)
    label_list.add(ENTITY_OUTSIDE_SYMBOL)
    # needs deterministic order for consistency
    return sorted(list(label_list))


def get_label_list_from_model(model_config_path: PathLike) -> list[str]:
    with open(model_config_path, "r") as file:
        model_config = json.load(file)
        id2label = {int(idx): label for idx, label in model_config["id2label"].items()}
        label_list = [label for _, label in sorted(id2label.items())]
    return label_list


def create_wrapper(cfg: DictConfig, label_list: list[str]) -> Optional[LSManagerViewWrapper]:
    if cfg.get("label_studio_manager") and cfg.get("css_colors"):
        ls_manager: LabelStudioManager = instantiate(cfg.label_studio_manager)
        css_colors = cfg.css_colors
        label_to_color = {}
        for i, label in enumerate(label_list):
            label_to_color[label] = css_colors[i]
        view = LabelStudioAnnotationView(ner_labels=label_to_color)
        return LSManagerViewWrapper(view, ls_manager)
    return None
