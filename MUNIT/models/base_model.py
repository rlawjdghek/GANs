from abc import ABC, abstractmethod
import torch.nn as nn
import torch
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_gene_label=0.0):
        super().__init__()
        self.real_label = target_real_label
        self.gene_label = target_gene_label
        if use_lsgan:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCELoss()
    def forward(self, x, target_is_real):
        '''
        x는 여러개의 NLayerD로 이루어진 MultiscaleD에서 나옴. 리스트안에 리스트가 있으므로 마지막 prediction만 사용해야한다. 
        '''
        if isinstance(x[0], list):
            loss = 0
            for i in x:
                pred = i[-1]
                if target_is_real:
                    label = torch.zeros(pred.shape).fill_(self.real_label).cuda(pred.get_device())
                else:
                    label = torch.zeros(pred.shape).fill_(self.gene_label).cuda(pred.get_device())
                loss += self.criterion(pred, label)
        else:
            pred = x[-1]
            if target_is_real:
                label = torch.zeros(pred.shape).fill_(self.real_label).cuda(pred.get_device())
            else:
                label = torch.zeros(pred.shape).fill_(self.gene_label).cuda(pred.get_device())
            loss = self.criterion(pred, label)
        return loss
# class GANLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x, target_is_real):
#         if target_is_real: target = 1.0
#         else: target = 0.0
#         loss = sum([torch.mean((out-target)**2) for out in x])
#         return loss

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
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