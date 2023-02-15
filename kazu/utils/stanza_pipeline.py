from functools import cache

import stanza
from stanza.pipeline.core import DownloadMethod

from kazu.utils.utils import PathLike


@cache
def simple_stanza_init(path: PathLike, download: bool, use_gpu: bool) -> stanza.Pipeline:
    if download:
        stanza.download(lang="en", package="genia", model_dir=path)
    stanza_pipeline = stanza.Pipeline(
        lang="en",
        model_dir=str(path),
        package=None,
        processors={"tokenize": "genia"},
        use_gpu=use_gpu,
        download_method=DownloadMethod.REUSE_RESOURCES,
    )
    return stanza_pipeline
