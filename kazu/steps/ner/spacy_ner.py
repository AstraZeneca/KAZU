from collections import defaultdict

from kazu.data.data import Document, Entity
from kazu.steps import Step, document_iterating_step

import spacy


class SpacyNerStep(Step):
    """
    A simple spacy NER implementation. Runs a spacy pipeline over document sections, expecting the
    resulting spacy doc to have a populated doc.ents field.
    """

    def __init__(self, spacy_pipeline: spacy.Language):
        """

        :param model_name: name of spacy pipeline to load.
        :param path: If the spacy pipeline is not already installed into the python environment, attempt to
            install it from this path.
        """
        self.nlp = spacy_pipeline

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        for section in doc.sections:
            spacy_doc = self.nlp(section.text)
            for ent in spacy_doc.ents:
                section.entities.append(
                    Entity.load_contiguous_entity(
                        start=ent.start_char,
                        end=ent.end_char,
                        match=section.text[ent.start_char : ent.end_char],
                        entity_class=ent.label_.lower(),
                        namespace=self.namespace(),
                    )
                )

            sent_metadata = defaultdict(list)
            for sent in spacy_doc.sents:
                sent_metadata["scispacy_sent_offsets"].append([sent.start_char, sent.end_char])
            if not section.metadata:
                section.metadata = {}
            section.metadata.update(sent_metadata)
