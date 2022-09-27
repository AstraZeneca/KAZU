import os

import pytest

from kazu.data.data import Document, Entity, Mapping, LinkRanks
from kazu.modelling.annotation.label_studio import LabelStudioManager, KazuToLabelStudioConverter
from kazu.tests.utils import requires_label_studio
from kazu.utils.grouping import sort_then_group


def create_test_manager():
    label_studio_url_and_port = os.environ["LS_URL_PORT"]
    headers = {
        "Authorization": f"Token {os.environ['LS_TOKEN']}",
        "Content-Type": "application/json",
    }
    ls_project = "kazu_integration_test"
    return LabelStudioManager(
        project_name=ls_project, headers=headers, url=label_studio_url_and_port
    )


@requires_label_studio
def test_kau_doc_to_label_studio():
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
    doc_1.sections[0].entities.append(e1)
    doc_1.sections[0].entities.append(e2)
    doc_1.sections[0].entities.append(e3)

    manager = create_test_manager()
    manager.delete_project_if_exists()
    tasks = KazuToLabelStudioConverter.convert_docs_to_tasks([doc_1])
    manager.create_linking_project(tasks)

    docs = manager.export_from_ls()
    assert len(docs) == 1
    doc = docs[0]
    for ent_class, ents_iter in sort_then_group(
        doc.get_entities(), key_func=lambda x: x.entity_class
    ):
        ents = list(ents_iter)
        assert len(ents) == 1

        mapping_sources = set(mapping.source for mapping in ents[0].mappings)
        mapping_ids = set(mapping.idx for mapping in ents[0].mappings)
        if ent_class == "gene":
            assert ents[0].match == "cat mat"
            assert len(mapping_sources) == 2
            assert "test1" in mapping_sources
            assert "test2" in mapping_sources
            assert "1" in mapping_ids
            assert "2" in mapping_ids
        elif ent_class == "disease":
            assert ents[0].match == "mat"
            assert len(mapping_sources) == 1
            assert "test3" in mapping_sources
            assert "3" in mapping_ids
        elif ent_class == "drug":
            assert ents[0].match == "cat"
            assert len(mapping_sources) == 0
            assert len(mapping_ids) == 0
        else:
            pytest.fail(f"{ent_class} should not exist")

    manager.delete_project_if_exists()
