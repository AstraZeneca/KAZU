from functools import cache

try:
    import stanza
    from stanza.pipeline.core import DownloadMethod
except ImportError as e:
    raise ImportError(
        "To use a stanza pipeline, you need to install stanza.\n"
        "You can either install stanza yourself, or install kazu[all-steps].\n"
    ) from e

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
