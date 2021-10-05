from omegaconf import DictConfig


class AZNerRunner:
    def __init__(self,cfg:DictConfig):
        self.cfg = cfg

    def ner(self,text:str)->str:
        return "not implemented"