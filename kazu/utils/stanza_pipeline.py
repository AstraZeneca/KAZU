import logging
import stanza
from stanza.pipeline.core import DownloadMethod

from typing import Optional

from kazu.utils.utils import PathLike

logger = logging.getLogger(__name__)


class StanzaPipeline:

    __instance: Optional[stanza.Pipeline] = None

    def __init__(self, stanza_nlp: stanza.Pipeline):
        if not StanzaPipeline.__instance:
            StanzaPipeline.__instance = stanza_nlp

    @property
    def instance(self) -> stanza.Pipeline:
        if self.__instance:
            return self.__instance
        else:
            raise NameError("stanza pipeline not initialised")

    @classmethod
    def from_stanza_kwargs(cls, **kwargs) -> "StanzaPipeline":
        stanza_pipeline = stanza.Pipeline(**kwargs)
        return StanzaPipeline(stanza_pipeline)

    @classmethod
    def simple_init(cls, path: PathLike, download: bool) -> "StanzaPipeline":
        if download:
            stanza.download(lang="en", package="genia", model_dir=path)

        stanza_pipeline = stanza.Pipeline(
            lang="en",
            model_dir=str(path),
            package=None,
            processors={"tokenize": "genia"},
            use_gpu=False,
            download_method=DownloadMethod.REUSE_RESOURCES,
        )
        return StanzaPipeline(stanza_pipeline)
