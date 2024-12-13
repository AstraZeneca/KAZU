import copy
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Union

from hydra.utils import instantiate
from omegaconf import DictConfig

from kazu.annotation.label_studio import (
    LabelStudioAnnotationView,
    LabelStudioManager,
)
from kazu.data import (
    ENTITY_OUTSIDE_SYMBOL,
    PROCESSING_EXCEPTION,
    Document,
    Entity,
    Section,
)
from kazu.utils.utils import PathLike

logger = logging.getLogger(__name__)


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


def chunks(lst: list[Any], n: int) -> Iterable[list[Any]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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


class LSManagerViewWrapper:
    def __init__(self, view: LabelStudioAnnotationView, ls_manager: LabelStudioManager):
        self.ls_manager = ls_manager
        self.view = view

    def get_gold_ents_for_side_by_side_view(self, docs: list[Document]) -> list[list[Document]]:
        result = []
        for doc in docs:
            doc_cp = copy.deepcopy(doc)
            if PROCESSING_EXCEPTION in doc_cp.metadata:
                logger.error(doc.metadata[PROCESSING_EXCEPTION])
                break
            for section in doc_cp.sections:
                gold_ents = []
                for ent in section.metadata.get("gold_entities", []):
                    if isinstance(ent, dict):
                        ent = Entity.from_dict(ent)
                    gold_ents.append(ent)
                section.entities = gold_ents
            result.append([doc_cp, doc])
        return result

    def update(
        self, docs: list[Document], global_step: Union[int, str], has_gs: bool = True
    ) -> None:
        ls_manager = LabelStudioManager(
            headers=self.ls_manager.headers,
            project_name=f"{self.ls_manager.project_name}_test_{global_step}",
        )
        ls_manager.delete_project_if_exists()
        ls_manager.create_linking_project()
        if not docs:
            logger.info("no results to represent yet")
            return
        if has_gs:
            side_by_side = self.get_gold_ents_for_side_by_side_view(docs)
            ls_manager.update_view(self.view, side_by_side)
            ls_manager.update_tasks(side_by_side)
        else:
            ls_manager.update_view(self.view, docs)
            ls_manager.update_tasks(docs)


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
