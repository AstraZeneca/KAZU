
from hydra.utils import instantiate

from kazu.data.data import Entity
from kazu.steps.ner.entity_post_processing import (
    NonContiguousEntitySplitter,
    SplitOnNumericalListPatternWithPrefix,
    SplitOnConjunctionPattern,
)
from kazu.tests.utils import requires_model_pack

simple_conjunction_example = "skin, lung and breast cancer are common forms."
complex_conjunction_example = "skin, lung and triple negative breast cancer are common forms."
numerical_slash_example = "everyone knows BRCA1/2/3 are oncogenes"
numerical_slash_example2 = "Monoclonal antibody D8/17 is a thing"


def check_expected_matches_are_in_entities(entities: list[Entity], expected_matches: set[str]):
    for ent in entities:
        if ent.match in expected_matches:
            expected_matches.discard(ent.match)
    assert not expected_matches


@requires_model_pack
def test_non_contiguous_entity_splitter(kazu_test_config):
    spacy_pipeline = instantiate(kazu_test_config.SpacyPipeline)

    splitter = NonContiguousEntitySplitter(
        entity_conditions={
            "gene": [SplitOnNumericalListPatternWithPrefix("/")],
            "disease": [SplitOnConjunctionPattern(spacy_pipeline)],
        }
    )

    numerical_slash_ent = Entity.load_contiguous_entity(
        start=15, end=24, namespace="test", entity_class="gene", match="BRCA1/2/3"
    )

    ents = splitter(numerical_slash_ent, numerical_slash_example)
    check_expected_matches_are_in_entities(
        entities=ents, expected_matches={"BRCA1", "BRCA2", "BRCA3"}
    )

    numerical_slash_ent = Entity.load_contiguous_entity(
        start=0, end=24, namespace="test", entity_class="gene", match="Monoclonal antibody D8/17"
    )

    ents = splitter(numerical_slash_ent, numerical_slash_example2)
    check_expected_matches_are_in_entities(
        entities=ents, expected_matches={"Monoclonal antibody D8", "Monoclonal antibody D17"}
    )

    conjunction_ent = Entity.load_contiguous_entity(
        start=0,
        end=27,
        namespace="test",
        entity_class="disease",
        match="skin, lung and breast cancer",
    )

    ents = splitter(conjunction_ent, simple_conjunction_example)
    check_expected_matches_are_in_entities(
        entities=ents, expected_matches={"skin cancer", "lung cancer", "breast cancer"}
    )

    conjunction_ent = Entity.load_contiguous_entity(
        start=0,
        end=44,
        namespace="test",
        entity_class="disease",
        match="skin, lung and triple negative breast cancer",
    )

    ents = splitter(conjunction_ent, complex_conjunction_example)
    check_expected_matches_are_in_entities(
        entities=ents,
        expected_matches={"skin cancer", "lung cancer", "triple negative breast cancer"},
    )
