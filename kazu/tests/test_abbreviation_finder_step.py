import pytest

from kazu.data.data import Section, Entity, Document
from kazu.steps.document_post_processing.abbreviation_finder import AbbreviationFinderStep


def test_AbbreviationFinderStep_copy_ents():
    sec1 = Section(
        text="Acute Mylenoid Leukaemia (AML) is a form of cancer. AML is treatable.", name="part1"
    )
    ent1 = Entity.load_contiguous_entity(
        match="Acute Mylenoid Leukaemia", entity_class="disease", start=0, end=24, namespace="test"
    )
    ent2 = Entity.load_contiguous_entity(
        match="AML", entity_class="gene", start=26, end=29, namespace="test"
    )
    sec1.entities = [ent1, ent2]
    sec2 = Section(text="AML is a serious disease", name="part1")
    ent3 = Entity.load_contiguous_entity(
        match="AML", entity_class="gene", start=0, end=3, namespace="test"
    )
    sec2.entities = [ent3]
    doc = Document(idx="test copy of entity data", sections=[sec1, sec2])

    step = AbbreviationFinderStep()

    success, failure = step([doc])
    assert len(failure) == 0

    assert len(sec1.entities) == 3
    assert len(sec2.entities) == 1

    for ent in doc.get_entities():
        assert ent.match in {"AML", "Acute Mylenoid Leukaemia"}
        assert ent.entity_class == "disease"

    sec3 = Section(text="Auto Mega Liquid (AML) is not form of cancer", name="part1")
    ent4 = Entity.load_contiguous_entity(
        match="AML", entity_class="disease", start=18, end=21, namespace="test"
    )
    sec3.entities.append(ent4)
    sec4 = Section(text="AML something I just made up", name="part1")
    ent5 = Entity.load_contiguous_entity(
        match="AML", entity_class="gene", start=0, end=3, namespace="test"
    )
    sec4.entities.append(ent5)
    doc = Document(idx="test removal of entity data", sections=[sec3, sec4])

    success, failure = step([doc])
    assert len(failure) == 0

    ents = doc.get_entities()
    assert len(ents) == 0


@pytest.mark.parametrize(argnames=("exclude_abbr"), argvalues=(([]), (["AML"])))
def test_AbbreviationFinderStep_remove_ents(exclude_abbr):

    step = AbbreviationFinderStep(exclude_abbrvs=exclude_abbr)

    sec3 = Section(text="Auto Mega Liquid (AML) is not form of cancer", name="part1")
    ent4 = Entity.load_contiguous_entity(
        match="AML", entity_class="disease", start=18, end=21, namespace="test"
    )
    sec3.entities.append(ent4)
    sec4 = Section(text="AML something I just made up", name="part1")
    ent5 = Entity.load_contiguous_entity(
        match="AML", entity_class="gene", start=0, end=3, namespace="test"
    )
    sec4.entities.append(ent5)
    doc = Document(idx="test removal of entity data", sections=[sec3, sec4])
    success, failure = step([doc])
    assert len(failure) == 0
    ents = doc.get_entities()
    if len(exclude_abbr) == 0:
        assert len(ents) == 0
    else:
        assert len(ents) == 2
        for ent in ents:
            assert ent.match == "AML"
            assert ent.entity_class in {"disease", "gene"}
