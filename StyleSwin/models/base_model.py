from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
    @abstractmethod
    def train(self): pass
    @abstractmethod
    def inference(self): pass
    @abstractmethod
    def load(self): pass
    @abstractmethod
    def save(self): pass
    @staticmethod
    def set_requires_grad(models, requires_grad=False):
        if not isinstance(models, list):
            models = [models]
        for model in models:
            if model is not None:
                for param in model.parameters(): 
                    param.requires_grad = requires_grad
    @staticmethod
    def accumulate(model1, model2, decay=0.999):  # 보통 model1 = G_ema, model2 = G.module (DDP 들어오면 안된다.)
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())
        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1-decay)

