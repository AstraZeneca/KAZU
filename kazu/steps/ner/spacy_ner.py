from kazu.data import Document, Entity, CharSpan
from kazu.steps import Step, document_iterating_step
from kazu.utils.spacy_pipeline import SpacyPipelines


class SpacyNerStep(Step):
    """A simple spacy NER implementation.

    Runs a spacy pipeline over document sections, expecting the resulting spacy doc to
    have a populated doc.ents field.
    """

    def __init__(self, path: str, add_sentence_spans: bool = True):
        """

        :param path: path to the spacy pipeline to use.
        :param add_sentence_spans: If True, add sentence spans to the section.
        """
        self.add_sentence_spans = add_sentence_spans
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
            if self.add_sentence_spans:
                section.sentence_spans = [
                    CharSpan(sent.start_char, sent.end_char) for sent in spacy_doc.sents
                ]
