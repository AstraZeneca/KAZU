from functools import cache

import spacy

from kazu.utils.utils import PathLike


@cache
def cached_spacy_pipeline_load(path: PathLike) -> spacy.Language:
    return spacy.load(path)
