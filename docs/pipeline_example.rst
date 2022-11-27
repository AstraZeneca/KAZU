.. code-block:: python

    import hydra
    from hydra.utils import instantiate
    from omegaconf import DictConfig

    from kazu.pipeline import Pipeline
    from kazu.data.data import Document


    @hydra.main(config_path="conf", config_name="config")
    def run_docs(cfg: DictConfig) -> None:
        pipeline: Pipeline = instantiate(cfg.Pipeline)
        docs = [Document.create_simple_document(x) for x in ["doc 1 text", "doc 2 text etc"]]
        pipeline(docs)


    if __name__ == "__main__":
        run_docs()
