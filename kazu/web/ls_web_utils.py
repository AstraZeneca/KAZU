from typing import Any

from kazu.data import Document
from kazu.annotation.label_studio import (
    LabelStudioAnnotationView,
    KazuToLabelStudioConverter,
)

LSView = str
LSTask = dict[str, Any]


class LSWebUtils:
    def __init__(
        self, ls_view_generator: LabelStudioAnnotationView, ls_converter: KazuToLabelStudioConverter
    ):
        self.ls_view_generator = ls_view_generator
        self.ls_converter = ls_converter

    def kazu_doc_to_ls(self, doc: Document) -> tuple[LSView, list[LSTask]]:
        ls_tasks = list(self.ls_converter.convert_single_doc_to_tasks(doc))
        ls_view = self.ls_view_generator.create_main_view(ls_tasks)
        return ls_view, ls_tasks

    @staticmethod
    def default() -> "LSWebUtils":
        return LSWebUtils(
            LabelStudioAnnotationView.with_default_colours(), KazuToLabelStudioConverter()
        )
