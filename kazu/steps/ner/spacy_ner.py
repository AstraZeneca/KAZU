import traceback
from collections import defaultdict
from typing import List, Tuple

from kazu.data.data import Document, PROCESSING_EXCEPTION, Entity
from kazu.steps import Step
from kazu.utils.spacy_pipeline import SpacyPipeline


class SpacyNerStep(Step):
    """
    A simple spacy NER implementation. Runs a spacy pipeline over document sections, expecting the
    resulting spacy doc to have a populated doc.ents field.
    """

    def __init__(self, depends_on: List[str], spacy_pipeline: SpacyPipeline):
        """

        :param depends_on:
        :param model_name: name of spacy pipeline to load.
        :param path: If the spacy pipeline is not already installed into the python environment, attempt to
            install it from this path.
        """
        super().__init__(depends_on=depends_on)
        self.nlp = spacy_pipeline.nlp

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        for doc in docs:
            try:
                for section in doc.sections:
                    spacy_doc = self.nlp(section.get_text())
                    for ent in spacy_doc.ents:
                        section.entities.append(
                            Entity.load_contiguous_entity(
                                start=ent.start_char,
                                end=ent.end_char,
                                match=section.get_text()[ent.start_char : ent.end_char],
                                entity_class=ent.label_.lower(),
                                namespace=self.namespace(),
                            )
                        )

                    sent_metadata = defaultdict(list)
                    for sent in spacy_doc.sents:
                        sent_metadata["scispacy_sent_offsets"].append(
                            [sent.start_char, sent.end_char]
                        )
                    if not section.metadata:
                        section.metadata = {}
                    section.metadata.update(sent_metadata)
            except Exception:
                message = f"doc failed: affected ids: {doc.idx}\n" + traceback.format_exc()
                doc.metadata[PROCESSING_EXCEPTION] = message
                failed_docs.append(doc)
                # docs.remove(doc)
        return docs, failed_docs
