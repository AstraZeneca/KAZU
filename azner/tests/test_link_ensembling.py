import pytest

from azner.steps.linking.link_ensembling import EnsembleEntityLinkingStep
from data.data import SimpleDocument, Entity, Mapping, LINK_SCORE


@pytest.mark.parametrize("keep_top_n", [1, 3])
def test_highest_score_link_ensembling(keep_top_n):
    step = EnsembleEntityLinkingStep([], keep_top_n=keep_top_n, method="highest_score")
    doc = SimpleDocument("hello")
    entity = Entity(namespace="test", start=0, end=1, match="hello", entity_class="test")

    mappings = [
        Mapping(source="test", idx="bad_match", mapping_type="direct", metadata={LINK_SCORE: 0.5}),
        Mapping(source="test", idx="ok_match", mapping_type="direct", metadata={LINK_SCORE: 0.8}),
        Mapping(
            source="test", idx="great_match", mapping_type="direct", metadata={LINK_SCORE: 1.0}
        ),
    ]
    entity.metadata.mappings = mappings
    doc.sections[0].entities = [entity]

    result, _ = step([doc])
    result_entities = result[0].get_entities()
    result_mappings = result_entities[0].metadata.mappings
    assert len(result_mappings) == keep_top_n
    if keep_top_n == 1:
        assert result_mappings[0].metadata[LINK_SCORE] == 1.0
    elif keep_top_n == 3:
        assert result_mappings[0].metadata[LINK_SCORE] == 1.0
        assert result_mappings[1].metadata[LINK_SCORE] == 0.8
        assert result_mappings[2].metadata[LINK_SCORE] == 0.5
