from typing import List, Dict

from hydra.utils import instantiate

from kazu.modelling.ontology_preprocessing.synonym_generation import (
    SeparatorExpansion,
    SynonymData,
    SynonymGenerator,
    CaseModifier,
    StopWordRemover,
    StringReplacement,
    GreekSymbolSubstitution,
)
from kazu.tests.utils import requires_model_pack


def check_generator_result(
    expected_syns: List[str], generator: SynonymGenerator, input: Dict[str, List[SynonymData]]
):
    result = generator(input)
    new_syns = list(result.keys())
    assert len(new_syns) == len(expected_syns)
    assert all([x in new_syns for x in expected_syns])


@requires_model_pack
def test_SeparatorExpansion(kazu_test_config):
    spacy_pipeline = instantiate(kazu_test_config.SpacyPipeline)
    generator = SeparatorExpansion(spacy_pipeline)
    input = {"ABAC (ABAC1/ABAC2)": [SynonymData(idx="text", mapping_type=[])]}
    expected_syns = ["ABAC", "ABAC1", "ABAC2", "ABAC1/ABAC2", "ABAC ABAC1/ABAC2"]
    check_generator_result(expected_syns=expected_syns, generator=generator, input=input)


def test_CaseModifier():
    generator = CaseModifier(lower=True)
    input = {"ABAC": [SynonymData(idx="text", mapping_type=[])]}
    expected_syns = ["abac"]
    check_generator_result(expected_syns=expected_syns, generator=generator, input=input)


@requires_model_pack
def test_StopWordRemover(kazu_test_config):
    spacy_pipeline = instantiate(kazu_test_config.SpacyPipeline)
    generator = StopWordRemover(spacy_pipeline=spacy_pipeline)
    input = {"The cat sat on the mat": [SynonymData(idx="text", mapping_type=[])]}
    expected_syns = ["cat sat mat"]
    check_generator_result(expected_syns=expected_syns, generator=generator, input=input)


def test_StringReplacement():
    generator = StringReplacement(replacement_dict={"cat": ["dog", "chicken"]})
    input = {"The cat sat on the mat": [SynonymData(idx="text", mapping_type=[])]}
    expected_syns = ["The dog sat on the mat", "The chicken sat on the mat"]
    check_generator_result(expected_syns=expected_syns, generator=generator, input=input)


def test_GreekSymbolSubstitution():
    generator = GreekSymbolSubstitution()
    input = {"alpha-thalassaemia": [SynonymData(idx="text", mapping_type=[])]}
    expected_syns = ["α-thalassaemia"]
    check_generator_result(expected_syns=expected_syns, generator=generator, input=input)
    input = {"α-thalassaemia": [SynonymData(idx="text", mapping_type=[])]}
    expected_syns = ["alpha-thalassaemia", "a-thalassaemia"]
    check_generator_result(expected_syns=expected_syns, generator=generator, input=input)
    input = {"A-thalassaemia": [SynonymData(idx="text", mapping_type=[])]}
    expected_syns = []
    check_generator_result(expected_syns=expected_syns, generator=generator, input=input)
