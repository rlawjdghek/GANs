from abc import ABC, abstractmethod

import torch
import torch.nn as nn
def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)
class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()
    def forward(self, pred, is_target_real):
        pred = torch.sigmoid(pred)
        if is_target_real:
            label = torch.ones_like(pred).to(pred.device)
        else:
            label = torch.zeros_like(pred).to(pred.device)
        loss = self.criterion(pred, label)
        return loss
class R1RegLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, D_out, x_in):
        BS = x_in.shape[0]
        grad_D_out = torch.autograd.grad(outputs=D_out.sum(), inputs=x_in, create_graph=True, retain_graph=True, only_inputs=True)[0].pow(2)
        assert grad_D_out.shape == x_in.shape
        reg = 0.5 * grad_D_out.reshape(BS, -1).sum(1).mean(0)
        return reg
class BaseModel(ABC):
    def __init__(self, args):
        self.args = args
    @abstractmethod
    def set_input(self): pass
    @abstractmethod
    def train(self): pass
    @staticmethod
    def set_requires_grad(models, requires_grad=False):
        if not isinstance(models, list):
            models = [models]
        for model in models:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = requires_grad