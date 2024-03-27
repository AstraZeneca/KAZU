from collections import defaultdict

from kazu.data import Document, Entity
from kazu.steps import Step, document_iterating_step
from kazu.utils.spacy_pipeline import SpacyPipelines


class SpacyNerStep(Step):
    """A simple spacy NER implementation.

    Runs a spacy pipeline over document sections, expecting the resulting spacy doc to
    have a populated doc.ents field.
    """

    def __init__(self, path: str):
        """

        :param path: path to the spacy pipeline to use.
        """
        self.path = path
        self.spacy_pipelines = SpacyPipelines()
        self.spacy_pipelines.add_from_path(path, path)

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        for section in doc.sections:
            spacy_doc = self.spacy_pipelines.process_single(section.text, model_name=self.path)
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
