import unicodedata
from sys import maxunicode
from typing import Union
import pytest
from omegaconf import DictConfig

from kazu.data.data import EquivalentIdSet, EquivalentIdAggregationStrategy, SynonymTerm
from kazu.language.language_phenomena import GREEK_SUBS
from kazu.ontology_preprocessing.synonym_generation import (
    SeparatorExpansion,
    SynonymGenerator,
    StopWordRemover,
    StringReplacement,
    GreekSymbolSubstitution,
    CombinatorialSynonymGenerator,
)
from kazu.utils.spacy_pipeline import BASIC_PIPELINE_NAME, SpacyPipelines, basic_spacy_pipeline

# this is frozen so we only need to instantiate once
dummy_equiv_ids = EquivalentIdSet(
    ids_and_source=frozenset(
        (
            (
                "text",
                "text",
            ),
        ),
    )
)


def check_generator_result(
    input_str: str,
    expected_syns: set[str],
    generator: Union[CombinatorialSynonymGenerator, SynonymGenerator],
):

    term = {
        SynonymTerm(
            terms=frozenset([input_str]),
            term_norm="NA",
            is_symbolic=False,
            mapping_types=frozenset(),
            associated_id_sets=frozenset([dummy_equiv_ids]),
            parser_name="test",
            aggregated_by=EquivalentIdAggregationStrategy.CUSTOM,
        )
    }

    result: set[SynonymTerm] = generator(term)

    new_syns = set(term for synonym in result for term in synonym.terms)
    assert new_syns == expected_syns


@pytest.fixture(scope="session")
def separator_expansion_generator(kazu_test_config: DictConfig) -> SeparatorExpansion:
    SpacyPipelines().add_from_func(BASIC_PIPELINE_NAME, basic_spacy_pipeline)
    return SeparatorExpansion(BASIC_PIPELINE_NAME)


# fmt: off
@pytest.mark.parametrize(
    argnames=("input_str", "expected_syns"),
    argvalues=(
        (
            "ABAC (ABAC1/ABAC2)",
            {
                "ABAC",
                "ABAC1",
                "ABAC2",
                "ABAC1/ABAC2",
                "ABAC ABAC1/ABAC2"},
        ),
        (
            "cyclin-dependent kinase inhibitor 1B (p27, Kip1)",
            {
                "cyclin-dependent kinase inhibitor 1B",
                "p27",
                "Kip1",
                "p27, Kip1",
                "cyclin-dependent kinase inhibitor 1B p27, Kip1",
            },
        ),
        (
            "gonadotropin-releasing hormone (type 2) receptor 2",
            {"gonadotropin-releasing hormone receptor 2"},
        ),
        (
            "oxidase (cytochrome c) assembly 1-like",
            {"oxidase assembly 1-like"},
        ),
    ),
)
# fmt: on
def test_SeparatorExpansion(input_str, expected_syns, separator_expansion_generator):
    check_generator_result(input_str, expected_syns, separator_expansion_generator)


def test_StopWordRemover():
    generator = StopWordRemover()
    check_generator_result(
        input_str="The cat sat in the mat", expected_syns={"cat sat mat"}, generator=generator
    )


def test_StringReplacement():
    generator = StringReplacement(replacement_dict={"cat": ["dog", "chicken"]})
    check_generator_result(
        input_str="The cat sat on the mat",
        expected_syns={"The dog sat on the mat", "The chicken sat on the mat"},
        generator=generator,
    )


@pytest.fixture(scope="session")
def greek_symbol_generator() -> StringReplacement:
    return StringReplacement(include_greek=True)


@pytest.mark.parametrize(
    argnames=("input_str", "expected_syns"),
    argvalues=(
        (
            "alpha-thalassaemia",
            {"α-thalassaemia", "Α-thalassaemia"},
        ),
        (
            "α-thalassaemia",
            {"alpha-thalassaemia", "a-thalassaemia", "Α-thalassaemia"},
        ),
        (
            "A-thalassaemia",
            set(),
        ),
        pytest.param(
            "beta test",
            {
                "β test",
                "ϐ test",
                "Β test",
            },
            marks=pytest.mark.xfail(reason="'beta' also gets substring 'eta' replaced"),
        ),
        pytest.param(
            "alpha beta test",
            {
                "alpha β test",
                "alpha ϐ test",
                "alpha Β test",
                "α beta test",
                "α β test",
                "α ϐ test",
                "α Β test",
                "Α beta test",
                "Α β test",
                "Α ϐ test",
                "Α Β test",
            },
            marks=pytest.mark.xfail(reason="above problem with beta/eta"),
        ),
    ),
)
def test_GreekSymbolSubstitution(input_str, expected_syns, greek_symbol_generator):
    check_generator_result(input_str, expected_syns, greek_symbol_generator)


def test_greek_substitution_is_stripped():
    for k, val_set in GreekSymbolSubstitution.ALL_SUBS.items():
        assert k.strip() == k
        assert all(val.strip() == val for val in val_set)


@pytest.mark.xfail(
    reason="awkward casing behaviour where there are e.g. multiple uppercase theta's in unicode."
)
def test_greek_substitution_dict_casing():
    for k, val_set in GreekSymbolSubstitution.ALL_SUBS.items():
        for v in val_set:
            # if we substitute to a greek letter, we should sub
            # to both cases
            if v in GREEK_SUBS:
                if v.islower():
                    flipped_case_v = v.upper()
                elif v.isupper():
                    flipped_case_v = v.lower()
                assert flipped_case_v in GREEK_SUBS
                assert flipped_case_v == k or flipped_case_v in val_set


@pytest.mark.xfail(reason="haven't covered (or marked as exceptions) all unicode greek chars yet")
def test_greek_substitution_dict_uncode_variants():
    for i in range(maxunicode):
        char = chr(i)
        try:
            char_name_lower = unicodedata.name(char).lower()
            for greek_letter in GREEK_SUBS.values():
                assert greek_letter not in char_name_lower
        except ValueError:
            # character doesn't exist or has no name
            pass


def test_CombinatorialSynonymGenerator():
    generator = CombinatorialSynonymGenerator(
        [
            StringReplacement(include_greek=True),
            StringReplacement(replacement_dict={"-": [" ", "_"]}, include_greek=False),
        ]
    )
    check_generator_result(
        input_str="alpha-thalassaemia",
        expected_syns={
            "alpha thalassaemia",
            "alpha_thalassaemia",
            "α-thalassaemia",
            "α thalassaemia",
            "α_thalassaemia",
            "Α-thalassaemia",
            "Α thalassaemia",
            "Α_thalassaemia",
        },
        generator=generator,
    )
