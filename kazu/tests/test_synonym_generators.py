from typing import List, Union

from hydra.utils import instantiate

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
    expected_syns: List[str],
    generator: Union[CombinatorialSynonymGenerator, SynonymGenerator],
):
    data = {input_str: set((dummy_syn_data,))}
    result = generator(data)
    new_syns = list(result.keys())
    assert len(new_syns) == len(expected_syns)
    assert all([x in new_syns for x in expected_syns])


@requires_model_pack
def test_SeparatorExpansion(kazu_test_config):
    spacy_pipeline = instantiate(kazu_test_config.SpacyPipeline)
    generator = SeparatorExpansion(spacy_pipeline)

    check_generator_result(
        input_str="ABAC (ABAC1/ABAC2)",
        expected_syns=["ABAC", "ABAC1", "ABAC2", "ABAC1/ABAC2", "ABAC ABAC1/ABAC2"],
        generator=generator,
    )

    check_generator_result(
        input_str="cyclin-dependent kinase inhibitor 1B (p27, Kip1)",
        expected_syns=[
            "cyclin-dependent kinase inhibitor 1B",
            "p27",
            "Kip1",
            "p27, Kip1",
            "cyclin-dependent kinase inhibitor 1B p27, Kip1",
        ],
        generator=generator,
    )

    check_generator_result(
        input_str="gonadotropin-releasing hormone (type 2) receptor 2",
        expected_syns=["gonadotropin-releasing hormone receptor 2"],
        generator=generator,
    )

    check_generator_result(
        input_str="oxidase (cytochrome c) assembly 1-like",
        expected_syns=["oxidase assembly 1-like"],
        generator=generator,
    )


def test_CaseModifier():
    generator = CaseModifier(lower=True)
    check_generator_result(input_str="ABAC", expected_syns=["abac"], generator=generator)


@requires_model_pack
def test_StopWordRemover(kazu_test_config):
    spacy_pipeline = instantiate(kazu_test_config.SpacyPipeline)
    generator = StopWordRemover(spacy_pipeline=spacy_pipeline)
    check_generator_result(
        input_str="The cat sat on the mat", expected_syns=["cat sat mat"], generator=generator
    )


def test_StringReplacement():
    generator = StringReplacement(replacement_dict={"cat": ["dog", "chicken"]})
    check_generator_result(
        input_str="The cat sat on the mat",
        expected_syns=["The dog sat on the mat", "The chicken sat on the mat"],
        generator=generator,
    )


def test_GreekSymbolSubstitution():
    generator = StringReplacement(include_greek=True)
    check_generator_result(
        input_str="alpha-thalassaemia",
        expected_syns=["α-thalassaemia"],
        generator=generator,
    )
    check_generator_result(
        input_str="α-thalassaemia",
        expected_syns=["alpha-thalassaemia", "a-thalassaemia"],
        generator=generator,
    )
    check_generator_result(input_str="A-thalassaemia", expected_syns=[], generator=generator)


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
        expected_syns=[
            "alpha-thalassaemia",
            "Alpha thalassaemia",
            "Alpha_thalassaemia",
            "α-thalassaemia",
            "alpha thalassaemia",
            "alpha_thalassaemia",
            "α thalassaemia",
            "α_thalassaemia",
        ],
        generator=generator,
    )
