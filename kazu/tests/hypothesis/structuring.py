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
    _initialize_json_converter,
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


simply_serializable_types = (
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
    Entity,
    Section,
    Document,
)

entity_and_containers = (Entity, Section, Document)
all_serializable_types = simply_serializable_types + entity_and_containers

comparable_class_strategy = st.one_of(*(st.from_type(t) for t in all_serializable_types))


testing_json_converter = _initialize_json_converter(testing=True)


@given(instance=comparable_class_strategy)
@settings(max_examples=100 * len(all_serializable_types))
def test_comparable_class_round_trip_structuring(instance):
    """Custom max examples because really we want to test all the different types as
    much as a single 'normal' test, for which the default is 100."""
    converter = (
        testing_json_converter if type(instance) in entity_and_containers else _json_converter
    )
    d = converter.unstructure(instance)
    restructured = converter.structure(d, type(instance))
    assert instance == restructured


@given(doc=...)
def test_document_converter_testing_vs_production(doc: Document):
    """Ensure that the 'testing' cattrs converter only differs as expected.

    Just test Document, not Section or Entity, since Document contains Sections, which
    contain entities.
    """
    d = _json_converter.unstructure(doc)
    d2 = testing_json_converter.unstructure(doc)
    for sec in d2.get("sections", []):
        for ent in sec.get("entities", []):
            del ent["_id"]

    assert d == d2
