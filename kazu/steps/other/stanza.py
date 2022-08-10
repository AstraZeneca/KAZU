from typing import List, Tuple, Optional

from kazu.data.data import Document, CharSpan, PROCESSING_EXCEPTION
from kazu.steps.base.step import BaseStep
from kazu.utils.stanza_pipeline import StanzaPipeline
from stanza.models.common.doc import Sentence

import traceback


class StanzaStep(BaseStep):
    """
    Stanza step
    Currently used for just sentence-segmentation using a tokenizer trained on the genia treebank

    @article{zhang2021biomedical,
      title={Biomedical and clinical English model packages for the Stanza Python NLP library},
      author={Zhang, Yuhao and Zhang, Yuhui and Qi, Peng and Manning, Christopher D and Langlotz, Curtis P},
      journal={Journal of the American Medical Informatics Association},
      volume={28},
      number={9},
      pages={1892--1899},
      year={2021},
      publisher={Oxford University Press}
    }
    """

    def __init__(self, depends_on: Optional[List[str]], stanza_pipeline: StanzaPipeline):
        """

        :param depends_on:
        :param stanza_pipeline: singleton wrapping a stanza pipeline
        """
        super().__init__(depends_on)
        self.stanza_nlp = stanza_pipeline.instance

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
