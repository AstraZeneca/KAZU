from functools import cache

import spacy


@cache
def cached_spacy_pipeline_load(path: str) -> spacy.Language:
    return spacy.load(path)
