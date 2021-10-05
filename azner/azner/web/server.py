import logging
import hydra
import ray
from fastapi import FastAPI
from omegaconf import DictConfig
from pydantic import BaseModel
from ray import serve

from azner.runner.azner_runner import AZNerRunner
from azner.web.routes import AZNER

logger = logging.getLogger("ray")
app = FastAPI()

class Item(BaseModel):
    text: str

@serve.deployment(route_prefix='/api')
@serve.ingress(app)
class AZNerWebApp:
    """
    Web app to serve results
    """
    def __init__(self, azner_runner:AZNerRunner):
        """
        :param azner_runner: instance of AZNerRunner
        """
        self.azner_runner = azner_runner

    @app.get("/")
    def get(self):
        logger.info("received request to root /")
        return "Welcome to AZNER."

    @app.post(f"/{AZNER}")
    def ner(self,item:Item):
        logger.info(f"received request: {item.text}")
        return self.azner_runner.ner(text=item.text)


@hydra.main(config_path="../conf", config_name="config")
def start(cfg:DictConfig)->None:
    """
    deploy the web app to Ray Serve
    :param cfg: DictConfig from Hydra
    :return: None
    """
    # Connect to the running Ray cluster, or run as single node
    ray.init(address=cfg.ray.address, namespace="serve")
    # Bind on 0.0.0.0 to expose the HTTP server on external IPs.
    serve.start(detached=True, http_options={'host':"0.0.0.0","location":"EveryNode"})
    calculator = AZNerRunner(cfg)
    AZNerWebApp.deploy(calculator)




if __name__ == "__main__":
    start()