from kazu.data.data import Document, Entity, Mapping, LinkRanks
from kazu.modelling.annotation.label_studio import (
    LabelStudioManager,
    KazuToLabelStudioConverter,
    LabelStudioAnnotationView,
)
from kazu.tests.utils import requires_label_studio
from kazu.utils.grouping import sort_then_group


@requires_label_studio
def test_kazu_doc_to_label_studio(make_label_studio_manager):
    text = "the cat sat on the mat"
    doc_1 = Document.create_simple_document(text)
    e1 = Entity.from_spans(
        [(4, 7), (19, 22)], text=text, join_str=" ", namespace="test", entity_class="gene"
    )
    e2 = Entity.from_spans(
        [(19, 22)], text=text, join_str=" ", namespace="test", entity_class="disease"
    )
    e3 = Entity.from_spans([(4, 7)], text=text, join_str=" ", namespace="test", entity_class="drug")

    e1.mappings.add(
        Mapping(
            default_label="cat mat",
            source="test1",
            parser_name="test1",
            idx="1",
            mapping_strategy="test",
            disambiguation_strategy=None,
            confidence=LinkRanks.HIGHLY_LIKELY,
            metadata={},
        )
    )
    e1.mappings.add(
        Mapping(
            default_label="cat mat",
            source="test2",
            parser_name="test2",
            idx="2",
            mapping_strategy="test",
            disambiguation_strategy=None,
            confidence=LinkRanks.HIGHLY_LIKELY,
            metadata={},
        )
    )

    e2.mappings.add(
        Mapping(
            default_label="mat",
            source="test3",
            parser_name="test3",
            idx="3",
            mapping_strategy="test",
            disambiguation_strategy=None,
            confidence=LinkRanks.HIGHLY_LIKELY,
            metadata={},
        )
    )
    doc_1.sections[0].entities.extend((e1, e2, e3))

    manager: LabelStudioManager = make_label_studio_manager(project_name="kazu_integration_test")
    manager.delete_project_if_exists()
    tasks = KazuToLabelStudioConverter.convert_docs_to_tasks([doc_1])
    view = LabelStudioAnnotationView(
        ner_labels={
            "cell_line": "red",
            "cell_type": "darkblue",
            "disease": "orange",
            "drug": "yellow",
            "gene": "green",
            "species": "purple",
            "anatomy": "pink",
            "go_mf": "grey",
            "go_cc": "blue",
            "go_bp": "brown",
        }
    )
    manager.create_linking_project(tasks, view)

    docs = manager.export_from_ls()
    assert len(docs) == 1
    doc = docs[0]
    for ent_class, ents_iter in sort_then_group(
        doc.sections[0].metadata["gold_entities"], key_func=lambda x: x.entity_class
    ):
        ents = list(ents_iter)
        assert len(ents) == 1

        ent = ents[0]
        mapping_sources = set(mapping.source for mapping in ent.mappings)
        mapping_ids = set(mapping.idx for mapping in ent.mappings)
        if ent_class == "gene":
            assert ent.match == "cat mat"
            assert {"test1", "test2"} == mapping_sources
            assert {"1", "2"} == mapping_ids
        elif ent_class == "disease":
            assert ent.match == "mat"
            assert {"test3"} == mapping_sources
            assert {"3"} == mapping_ids
        elif ent_class == "drug":
            assert ent.match == "cat"
    manager.delete_project_if_exists()
