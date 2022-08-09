from typing import List, Tuple, Optional

from kazu.data.data import Document, CharSpan, PROCESSING_EXCEPTION
from kazu.steps.base.step import BaseStep
from kazu.utils.utils import as_path, PathLike

import stanza
from stanza.pipeline.core import DownloadMethod
from stanza.models.common.doc import Sentence

import traceback


class StanzaSentenceSegmentation(BaseStep):
    """
    Sentence segmentation step using a tokenizer trained on the genia treebank
    """

    def __init__(self, depends_on: Optional[List[str]], path: PathLike, build_pipeline: bool):
        """

        :param depends_on:
        :param path: path to this step's model storage
        :param build_pipeline: do a fresh build of the stanza pipeline as part of init., and download the GENIA tokenizer
        """
        # need to load up Stanza from model pack
        super().__init__(depends_on)
        self.stanza_nlp = (
            self.build_stanza_pipeline(path)
            if build_pipeline
            else self.load_stanza_pipline(as_path(path))
        )

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []

        for doc in docs:
            try:
                for section in doc.sections:
                    stanza_doc = self.stanza_nlp(section.get_text())
                    sentences: List[Sentence] = stanza_doc.sentences
                    char_spans = (
                        CharSpan(sent.tokens[0].start_char, sent.tokens[-1].end_char)
                        for sent in sentences
                    )
                    section.sentence_spans.extend(char_spans)
            except Exception:
                doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                failed_docs.append(doc)

        return docs, failed_docs

    @staticmethod
    def build_stanza_pipeline(model_dir: PathLike) -> stanza.Pipeline:
        stanza.download(lang="en", package="genia", model_dir=model_dir)
        return stanza.Pipeline(
            "en",
            model_dir=str(as_path(model_dir)),
            package=None,
            processors={"tokenize": "genia"},
            use_gpu=False,
            download_method=DownloadMethod.REUSE_RESOURCES,
        )

    @staticmethod
    def load_stanza_pipline(model_dir: PathLike) -> stanza.Pipeline:
        return stanza.Pipeline(
            "en",
            model_dir=str(as_path(model_dir)),
            package=None,
            processors={"tokenize": "genia"},
            use_gpu=False,
            download_method=DownloadMethod.REUSE_RESOURCES,
        )
