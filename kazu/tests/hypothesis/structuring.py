from hypothesis import given, settings, strategies as st
import bson

from kazu.data.data import (
    EquivalentIdSet,
    Mapping,
    SynonymTerm,
    SynonymTermWithMetrics,
    Entity,
    Section,
    Document,
    MentionForm,
    CuratedTerm,
    CharSpan,
    ParserAction,
    GlobalParserActions,
    ParserBehaviour,
    _json_converter,
)

hex_string_len_24_strat = st.text(
    alphabet={"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"},
    min_size=24,
    max_size=24,
)
oid_arg_strat = st.one_of(st.none(), st.binary(min_size=12, max_size=12), hex_string_len_24_strat)
oid_strat = st.builds(bson.ObjectId, oid=oid_arg_strat)
st.register_type_strategy(bson.ObjectId, oid_strat)

# needed, otherwise the __post_init__ throws an error in calc_starts_and_ends when the set of spans is empty.
st.register_type_strategy(
    Entity, st.builds(Entity, spans=st.frozensets(st.builds(CharSpan), min_size=1))
)

# needed, otherwise the __post_init__ throws an error for length 0 parser_to_target_id_mappings
# or if a set of the target_id_mappings is empty.
st.register_type_strategy(
    ParserAction,
    st.builds(
        ParserAction,
        parser_to_target_id_mappings=st.dictionaries(
            keys=st.text(),
            values=st.sets(elements=st.text(), min_size=1),
            min_size=1,
        ),
    ),
)


comparable_serializable_types = (
    CharSpan,
    EquivalentIdSet,
    Mapping,
    SynonymTerm,
    SynonymTermWithMetrics,
    MentionForm,
    CuratedTerm,
    ParserAction,
    GlobalParserActions,
    ParserBehaviour,
)
comparable_class_strategy = st.one_of(*(st.from_type(t) for t in comparable_serializable_types))


@given(instance=comparable_class_strategy)
@settings(max_examples=100 * len(comparable_serializable_types))
def test_comparable_class_round_trip_structuring(instance):
    """Custom max examples because really we want to test all the different types as
    much as a single 'normal' test, for which the default is 100."""
    d = _json_converter.unstructure(instance)
    restructured = _json_converter.structure(d, type(instance))
    assert instance == restructured


def compare_entities(e1: Entity, e2: Entity) -> bool:
    """`Entity` has a custom __eq__ so that only the exact same entities are equal to
    one another.

    This is for hashing behaviour when creating sets/dictionaries of entities so that behaviour is as expected
    and we don't accidentally 'merge' two entities that should be different just because they contain the same data.

    #### TODO: discuss with Richard - can we just add an `_id` field on Entity with a uuid default factory? If we
    #### did this, I think correct hashing/eq behaviour as is currently would just fall out, it's more visible, and
    #### the tests here can be simpler.

    Therefore to compare them, we have to compare their contents explicitly - using `__dict__` works.
    """
    return e1.__dict__ == e2.__dict__


@given(ent=...)
def test_round_trip_entity_structuring(ent: Entity):
    d = _json_converter.unstructure(ent)
    restructured = _json_converter.structure(d, Entity)
    assert compare_entities(ent, restructured)


@given(sec=...)
def test_round_trip_section_structuring(sec: Section):
    d = _json_converter.unstructure(sec)
    restructured = _json_converter.structure(d, Section)
    for e1, e2 in zip(sec.entities, restructured.entities):
        assert compare_entities(e1, e2)
    sec.entities = []
    restructured.entities = []
    assert sec == restructured


@given(doc=...)
def test_round_trip_document_structuring(doc: Document):
    d = doc.from_json(doc.to_json())
    for e1, e2 in zip(doc.get_entities(), d.get_entities()):
        assert compare_entities(e1, e2)

    for section in doc.sections:
        section.entities = []
    for section in d.sections:
        section.entities = []

    assert doc == d
