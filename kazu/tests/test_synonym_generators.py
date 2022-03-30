from typing import List, Dict, Union, Set

from hydra.utils import instantiate

from kazu.modelling.ontology_preprocessing.synonym_generation import (
    SeparatorExpansion,
    SynonymGenerator,
    CaseModifier,
    StopWordRemover,
    StringReplacement,
    GreekSymbolSubstitution,
    CombinatorialSynonymGenerator,
)
from kazu.data.data import SynonymData
from kazu.tests.utils import requires_model_pack


def check_generator_result(
    expected_syns: List[str],
    generator: Union[CombinatorialSynonymGenerator, SynonymGenerator],
    data: Dict[str, Set[SynonymData]],
):
    result = generator(data)
    new_syns = list(result.keys())
    assert len(new_syns) == len(expected_syns)
    assert all([x in new_syns for x in expected_syns])


@requires_model_pack
def test_SeparatorExpansion(kazu_test_config):
    spacy_pipeline = instantiate(kazu_test_config.SpacyPipeline)
    generator = SeparatorExpansion(spacy_pipeline)
    data = {"ABAC (ABAC1/ABAC2)": [SynonymData(ids=frozenset(["text"]), mapping_type=[])]}
    expected_syns = ["ABAC", "ABAC1", "ABAC2", "ABAC1/ABAC2", "ABAC ABAC1/ABAC2"]
    check_generator_result(expected_syns=expected_syns, generator=generator, data=data)
    data = {
        "cyclin-dependent kinase inhibitor 1B (p27, Kip1)": [
            SynonymData(ids=frozenset(["text"]), mapping_type=[])
        ]
    }
    expected_syns = [
        "cyclin-dependent kinase inhibitor 1B",
        "p27",
        "Kip1",
        "p27, Kip1",
        "cyclin-dependent kinase inhibitor 1B p27, Kip1",
    ]
    check_generator_result(expected_syns=expected_syns, generator=generator, data=data)

    data = {
        "gonadotropin-releasing hormone (type 2) receptor 2": [
            SynonymData(ids=frozenset(["text"]), mapping_type=[])
        ]
    }
    expected_syns = ["gonadotropin-releasing hormone receptor 2"]
    check_generator_result(expected_syns=expected_syns, generator=generator, data=data)

    data = {
        "oxidase (cytochrome c) assembly 1-like": [
            SynonymData(ids=frozenset(["text"]), mapping_type=[])
        ]
    }
    expected_syns = ["oxidase assembly 1-like"]
    check_generator_result(expected_syns=expected_syns, generator=generator, data=data)


def test_CaseModifier():
    generator = CaseModifier(lower=True)
    data = {"ABAC": [SynonymData(ids=frozenset(["text"]), mapping_type=[])]}
    expected_syns = ["abac"]
    check_generator_result(expected_syns=expected_syns, generator=generator, data=data)


@requires_model_pack
def test_StopWordRemover(kazu_test_config):
    spacy_pipeline = instantiate(kazu_test_config.SpacyPipeline)
    generator = StopWordRemover(spacy_pipeline=spacy_pipeline)
    data = {"The cat sat on the mat": [SynonymData(ids=frozenset(["text"]), mapping_type=[])]}
    expected_syns = ["cat sat mat"]
    check_generator_result(expected_syns=expected_syns, generator=generator, data=data)


def test_StringReplacement():
    generator = StringReplacement(replacement_dict={"cat": ["dog", "chicken"]})
    data = {"The cat sat on the mat": [SynonymData(ids=frozenset(["text"]), mapping_type=[])]}
    expected_syns = ["The dog sat on the mat", "The chicken sat on the mat"]
    check_generator_result(expected_syns=expected_syns, generator=generator, data=data)


def test_GreekSymbolSubstitution():
    generator = GreekSymbolSubstitution()
    data = {"alpha-thalassaemia": [SynonymData(ids=frozenset(["text"]), mapping_type=[])]}
    expected_syns = ["α-thalassaemia"]
    check_generator_result(expected_syns=expected_syns, generator=generator, data=data)
    data = {"α-thalassaemia": [SynonymData(ids=frozenset(["text"]), mapping_type=[])]}
    expected_syns = ["alpha-thalassaemia", "a-thalassaemia"]
    check_generator_result(expected_syns=expected_syns, generator=generator, data=data)
    data = {"A-thalassaemia": [SynonymData(ids=frozenset(["text"]), mapping_type=[])]}
    expected_syns = []
    check_generator_result(expected_syns=expected_syns, generator=generator, data=data)


def test_CombinatorialSynonymGenerator():
    generator = CombinatorialSynonymGenerator(
        [
            GreekSymbolSubstitution(),
            CaseModifier(lower=True),
            StringReplacement(replacement_dict={"-": [" ", "_"]}),
        ]
    )
    data = {"Alpha-thalassaemia": [SynonymData(ids=frozenset(["text"]), mapping_type=[])]}
    expected_syns = [
        "alpha-thalassaemia",
        "Alpha thalassaemia",
        "Alpha_thalassaemia",
        "α-thalassaemia",
        "alpha thalassaemia",
        "alpha_thalassaemia",
        "α thalassaemia",
        "α_thalassaemia",
    ]
    check_generator_result(expected_syns=expected_syns, generator=generator, data=data)
