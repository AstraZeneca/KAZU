import logging

import spacy

logger = logging.getLogger(__name__)


class SpacyPipeline:
    """
    Singleton of a spacy pipeline, so we can reuse it across steps without needing to load the model
    multiple times
    """

    instance = None

    class __SpacyPipeline:
        def __init__(self, path: str):
            self.nlp = spacy.load(path)

    def __init__(self, path: str):
        if not SpacyPipeline.instance:
            SpacyPipeline.instance = SpacyPipeline.__SpacyPipeline(path)

    def __getattr__(self, name):
        return getattr(self.instance, name)
