import re

from hydra.utils import instantiate

from kazu.data import Document, Section, CharSpan
from kazu.tests.utils import (
    requires_model_pack,
    ner_long_document_test_cases,
    maybe_skip_experimental_tests,
)
from kazu.utils.utils import sort_then_group


def _add_spans_to_docs(docs: list[Document]):
    for doc in docs:
        for section in doc.sections:
            spans = []
            start = 0

            for match in re.finditer(r"\.", section.text):
                end = match.start()
                spans.append(CharSpan(start, end))
                start = end + 1
            section.sentence_spans = spans


@requires_model_pack
def test_TransformersModelForTokenClassificationNerStep(kazu_test_config, ner_simple_test_cases):
    # note, here we just test that the step is functional. Model performance is tested via an acceptance test
    step = instantiate(kazu_test_config.TransformersModelForTokenClassificationNerStep)
    processed, failures = step(ner_simple_test_cases)
    assert len(processed) == len(ner_simple_test_cases)
    assert len(failures) == 0


class TestGLINER:
    @maybe_skip_experimental_tests
    @requires_model_pack
    def test_GLINERStep_majority_vote(self, gliner_step):
        drug_section = Section(text="abracodabravir is a drug.", name="test1")
        gene_section1 = Section(text="abracodabravir is a gene.", name="test2")
        gene_section2 = Section(text="abracodabravir is definitely a gene.", name="test3")
        conflicted_doc = Document(sections=[drug_section, gene_section1, gene_section2])
        _add_spans_to_docs([conflicted_doc])
        processed, failures = gliner_step([conflicted_doc])
        assert len(processed) == 1
        assert len(failures) == 0
        for ent_class, ents in sort_then_group(
            conflicted_doc.get_entities(), lambda x: x.entity_class
        ):
            assert ent_class == "gene", "non-gene entity types detected"
            assert len(list(ents)) == 3

    @maybe_skip_experimental_tests
    @requires_model_pack
    def test_GLINERStep_long_document(self, gliner_step):
        doc_string, mention_count, long_doc_ent_class = ner_long_document_test_cases()[0]
        long_docs = [Document.create_simple_document(doc_string)]
        _add_spans_to_docs(long_docs)
        processed, failures = gliner_step(long_docs)
        assert len(processed) == len(long_docs)
        assert len(failures) == 0

        entities_grouped = {
            spans: list(ents)
            for spans, ents in sort_then_group(long_docs[0].sections[0].entities, lambda x: x.spans)
        }
        assert len(entities_grouped) == mention_count
        for ent_list in entities_grouped.values():
            assert len(ent_list) == 1
            assert ent_list[0].entity_class == long_doc_ent_class
