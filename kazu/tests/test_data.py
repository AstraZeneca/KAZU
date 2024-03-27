from hypothesis import assume, given, settings, strategies as st
import bson
import pytest

from kazu.data import (
    EquivalentIdSet,
    Mapping,
    LinkingCandidate,
    LinkingMetrics,
    CandidatesToMetrics,
    Entity,
    Section,
    Document,
    Synonym,
    OntologyStringResource,
    CharSpan,
    ParserAction,
    GlobalParserActions,
    MentionConfidence,
    StringMatchConfidence,
    DisambiguationConfidence,
    EquivalentIdAggregationStrategy,
    OntologyStringBehaviour,
    ParserBehaviour,
    kazu_json_converter,
    _initialize_json_converter,
)
from kazu.tests.utils import make_dummy_linking_candidate


def test_overlap_logic():
    # e.g. "the patient has metastatic liver cancers"
    e1 = Entity(
        namespace="test",
        match="metastatic liver cancer",
        entity_class="test",
        spans=frozenset([CharSpan(start=16, end=39)]),
    )
    e2 = Entity(
        namespace="test",
        match="liver cancers",
        entity_class="test",
        spans=frozenset([CharSpan(start=27, end=40)]),
    )

    assert e1.is_partially_overlapped(e2)

    # e.g. 'liver and lung cancer'
    e1 = Entity(
        namespace="test",
        match="liver cancer",
        entity_class="test",
        spans=frozenset([CharSpan(start=0, end=4), CharSpan(start=15, end=21)]),
    )
    e2 = Entity(
        namespace="test",
        match="lung cancer",
        entity_class="test",
        spans=frozenset([CharSpan(start=9, end=21)]),
    )
    assert not e1.is_partially_overlapped(e2)


def test_candidate_manipulation():
    e1 = Entity(
        namespace="test",
        match="metastatic liver cancer",
        entity_class="test",
        spans=frozenset([CharSpan(start=16, end=39)]),
    )

    # first test candidates are merged correctly (same id set, same parser name)
    candidate_1, metrics = make_dummy_linking_candidate(
        ["1", "2", "3"], parser_name="test", search_score=99.5
    )
    e1.add_or_update_linking_candidate(candidate_1, metrics)
    candidate_2, metrics = make_dummy_linking_candidate(
        ["1", "2", "3"], parser_name="test", embed_score=99.6
    )
    e1.add_or_update_linking_candidate(candidate_2, metrics)

    assert len(e1.linking_candidates) == 1
    candidate, merged_metric = next(iter(e1.linking_candidates.items()))
    assert merged_metric.embed_score == 99.6
    assert merged_metric.search_score == 99.5

    # now test syn_terms are differentiated if parser name is different
    candidate_3, metrics = make_dummy_linking_candidate(
        ["1", "2", "3"], parser_name="test_2", search_score=99.5
    )
    e1.add_or_update_linking_candidate(candidate_3, metrics)
    assert len(e1.linking_candidates) == 2

    # now test syn_terms are differentiated if id set is different
    candidate_4, metrics = make_dummy_linking_candidate(
        ["1", "2"], parser_name="test", search_score=99.5
    )
    e1.add_or_update_linking_candidate(candidate_4, metrics)
    assert len(e1.linking_candidates) == 3


def test_section_sentence_spans_is_immutable():
    x = Document.create_simple_document("Hello")
    x.sections[0].sentence_spans = [CharSpan(start=0, end=28), CharSpan(start=29, end=50)]

    # try re-assigning sentence_spans, which should raise an error
    with pytest.raises(AttributeError):
        x.sections[0].sentence_spans = [CharSpan(start=0, end=28), CharSpan(start=29, end=50)]


hex_string_len_24_strat = st.text(
    alphabet={"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"},
    min_size=24,
    max_size=24,
)
oid_arg_strat = st.one_of(st.none(), st.binary(min_size=12, max_size=12), hex_string_len_24_strat)
oid_strat = st.builds(bson.ObjectId, oid=oid_arg_strat)
st.register_type_strategy(bson.ObjectId, oid_strat)

# needed, otherwise the __post_init__ throws an error in calc_starts_and_ends when the set of spans is empty.
valid_spans_for_entity = st.frozensets(st.builds(CharSpan), min_size=1)

st.register_type_strategy(
    Entity,
    st.builds(
        Entity, spans=valid_spans_for_entity, linking_candidates=st.from_type(CandidatesToMetrics)
    ),
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


@st.composite
def ontology_resource_strat_no_conflict(
    draw: st.DrawFn, _resource_class: type[OntologyStringResource], /, *args, **kwargs
) -> OntologyStringResource:
    """This strategy ensures no error is raised in OntologyStringResource.__post__init__
    .

    Ensuring `original_synonyms` is of min_size 1 is one key here, but the other
    is handling case conflicts. We do this by handling any ValueErrors and 'assuming' them
    away with hypothesis.

    One annoying side affect here is that when we 'register' this as a type strategy,
    for some reason it takes the class `OntologyStringResource` as the second argument, after `draw`.
    We don't need this in the function body, but we need to handle it, otherwise it gets
    passed through in the other args to st.builds (which we do want for flexibility).

    Also mypy/hypothesis' type hints wants the first two args to be positional only
    apparently, hence the `/`.
    """
    try:
        ct = draw(
            st.builds(
                OntologyStringResource,
                original_synonyms=st.frozensets(elements=st.builds(Synonym), min_size=1),
                *args,
                **kwargs,
            )
        )
        conflict = False
    except ValueError:
        conflict = True
    assume(not conflict)
    return ct


st.register_type_strategy(OntologyStringResource, ontology_resource_strat_no_conflict)


@st.composite
def add_sent_spans(draw, s: Section):
    sent_spans = draw(
        # min_size=1 needed because if you pass an empty list, the 'omit_default_values' behaviour
        # we've configured for cattrs means that it is restructured into None, so the test would fail,
        # but this behaviour is fine.
        st.one_of(st.none(), (st.lists(elements=st.builds(CharSpan), unique=True, min_size=1)))
    )
    if sent_spans is not None:
        s.sentence_spans = sent_spans
    return s


# needed, as _sentence_spans is `init=False`, so otherwise, it's always None
st.register_type_strategy(Section, st.builds(Section).flatmap(add_sent_spans))


simply_serializable_types = (
    CharSpan,
    EquivalentIdSet,
    Mapping,
    LinkingCandidate,
    LinkingMetrics,
    Synonym,
    OntologyStringResource,
    ParserAction,
    GlobalParserActions,
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
        testing_json_converter if type(instance) in entity_and_containers else kazu_json_converter
    )
    d = converter.unstructure(instance)
    restructured = converter.structure(d, type(instance))
    assert instance == restructured


# we could test these with the round trip test, but they're all included in other classes above, so testing those is sufficient
enums = (
    MentionConfidence,
    StringMatchConfidence,
    DisambiguationConfidence,
    EquivalentIdAggregationStrategy,
    OntologyStringBehaviour,
    ParserBehaviour,
)

# this doesn't cover everything imaginable, because it doesn't do e.g. where a value is itself a dictionary of strings to a list of dictionaries of....
# but it's enough to prove that the relevant classes are covered.
# We can actually do everything using st.recursive, but it makes the test take 7/8 seconds longer,
# and really we'll mostly testing cattrs at that point, not our code.
valid_json_metadata = st.dictionaries(
    keys=st.text(), values=st.one_of(*(st.from_type(t) for t in all_serializable_types + enums))
)

class_with_complex_metadata = st.one_of(
    st.builds(Mapping, metadata=valid_json_metadata),
    # spans as above - unfortunately we have to repeat it here.
    st.builds(Entity, metadata=valid_json_metadata, spans=valid_spans_for_entity),
    st.builds(Section, metadata=valid_json_metadata),
    st.builds(Document, metadata=valid_json_metadata),
)


@given(instance=class_with_complex_metadata)
@settings(max_examples=400)
def test_complex_metadata_roundtrip(instance):
    """Custom max examples because really we want to test all the different types as
    much as a single 'normal' test, for which the default is 100.

    Note that we test that we can roundtrip, but don't assert that they're the same
    here: that's because cattrs doesn't know how to handle the untyped metadata: dict
    by default, so you just get dicts back when structuring. See the discussion
    in `docs/datamodel.rst`.
    """
    d = kazu_json_converter.unstructure(instance)
    kazu_json_converter.structure(d, type(instance))


@given(...)
def test_document_converter_testing_vs_production(doc: Document):
    """Ensure that the 'testing' cattrs converter only differs as expected.

    Just test Document, not Section or Entity, since Document contains Sections, which
    contain entities.
    """
    d = kazu_json_converter.unstructure(doc)
    d2 = testing_json_converter.unstructure(doc)
    for sec in d2.get("sections", []):
        for ent in sec.get("entities", []):
            del ent["_id"]

    assert d == d2
