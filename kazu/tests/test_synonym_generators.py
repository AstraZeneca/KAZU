from typing import Set, Union

from hydra.utils import instantiate
import pytest

from kazu.modelling.ontology_preprocessing.synonym_generation import (
    SeparatorExpansion,
    SynonymGenerator,
    CaseModifier,
    StopWordRemover,
    StringReplacement,
    CombinatorialSynonymGenerator,
)
from kazu.data.data import SynonymData, EquivalentIdAggregationStrategy
from kazu.tests.utils import requires_model_pack

# this is frozen so we only need to instantiate once
dummy_syn_data = SynonymData(
    ids=frozenset(("text",)),
    mapping_type=frozenset(),
    aggregated_by=EquivalentIdAggregationStrategy.UNAMBIGUOUS,
)


def check_generator_result(
    input_str: str,
    expected_syns: Set[str],
    generator: Union[CombinatorialSynonymGenerator, SynonymGenerator],
):
    data = {input_str: set((dummy_syn_data,))}
    result = generator(data)
    new_syns = set(result.keys())
    assert new_syns == expected_syns


@pytest.fixture(scope="session")
def separator_expansion_generator(kazu_test_config) -> SeparatorExpansion:
    spacy_pipeline = instantiate(kazu_test_config.SpacyPipeline)
    return SeparatorExpansion(spacy_pipeline)


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
@requires_model_pack
def test_SeparatorExpansion(input_str, expected_syns, separator_expansion_generator):
    check_generator_result(input_str, expected_syns, separator_expansion_generator)


def test_CaseModifier():
    generator = CaseModifier(lower=True)
    check_generator_result(input_str="ABAC", expected_syns={"abac"}, generator=generator)


@requires_model_pack
def test_StopWordRemover(kazu_test_config):
    spacy_pipeline = instantiate(kazu_test_config.SpacyPipeline)
    generator = StopWordRemover(spacy_pipeline=spacy_pipeline)
    check_generator_result(
        input_str="The cat sat on the mat", expected_syns={"cat sat mat"}, generator=generator
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
    ),
)
def test_GreekSymbolSubstitution(input_str, expected_syns, greek_symbol_generator):
    check_generator_result(input_str, expected_syns, greek_symbol_generator)


def test_CombinatorialSynonymGenerator():
    generator = CombinatorialSynonymGenerator(
        [
            StringReplacement(include_greek=True),
            CaseModifier(lower=True),
            StringReplacement(replacement_dict={"-": [" ", "_"]}),
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
