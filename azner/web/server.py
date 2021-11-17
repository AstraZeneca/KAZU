import logging
import time

import hydra
import ray
from fastapi import FastAPI
from omegaconf import DictConfig
from ray import serve
from azner.web.routes import AZNER
from azner.data.data import SimpleDocument
from azner.pipeline.pipeline import Pipeline, load_steps

logger = logging.getLogger("ray")
app = FastAPI()


@serve.deployment(route_prefix="/api")
@serve.ingress(app)
class AZNerWebApp:
    """
    Web app to serve results
    """

    def __init__(self, cfg: DictConfig):
        """
        :param azner_runner: instance of PipelineRunner
        """
        self.pipeline = Pipeline(load_steps(cfg))

    @app.get("/")
    def get(self):
        logger.info("received request to root /")
        return "Welcome to AZNER."

    @app.post(f"/{AZNER}")
    def ner(self, doc: SimpleDocument):
        logger.info(f"received request: {doc}")
        result = self.pipeline([doc])
        return result[0]


@hydra.main(config_path="../conf", config_name="config")
def start(cfg: DictConfig) -> None:
    """
    deploy the web app to Ray Serve
    :param cfg: DictConfig from Hydra
    :return: None
    """
    # Connect to the running Ray cluster, or run as single node
    ray.init(address=cfg.ray.address, namespace="serve")
    # Bind on 0.0.0.0 to expose the HTTP server on external IPs.
    serve.start(
        detached=cfg.ray.detached, http_options={"host": "0.0.0.0", "location": "EveryNode"}
    )
    AZNerWebApp.deploy(cfg)
    if not cfg.ray.detached:
        while True:
            logger.info(serve.list_deployments())
            time.sleep(10)


if __name__ == "__main__":
    start()
