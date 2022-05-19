from abc import ABC, abstractmethod
import torch.nn.functional as F
import torch

def D_reshape(tensors):  # 맨처음엔 5차원 텐서들 담긴 리스트 오는데 모두 4차원으로 바꿔서 다시 리스트로 return
    if isinstance(tensors, list):
        return [D_reshape(t) for t in tensors]
    BS, T, C, H, W = tensors.shape
    return tensors.contiguous().view(-1, C, H, W)
def get_grid(BS, H, W, device):  # (x,y)좌표를 HxW만큼 찍는다. 단, 범위는 -1~1 왼쪽 위가 (-1,-1)
    hor = torch.linspace(-1.0, 1.0, W)
    hor.requires_grad = False
    hor = hor.reshape(1,1,1,W)
    hor = hor.expand(BS, 1, H, W)
    ver = torch.linspace(-1.0, 1.0, H)
    ver.requires_grad = False
    ver = ver.reshape(1,1,H,1)
    ver = ver.expand(BS, 1, H, W)

    t_grid = torch.cat([hor, ver], 1)
    t_grid.requires_grad = False
    return t_grid.cuda(device)
class BaseModel(ABC):
    def __init__(self, args):
        self.args = args
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
    def grid_sample(self, input1, input2):
        return F.grid_sample(input1, input2, mode="bilinear", padding_mode="border")
    def resample(self, img, flow):
        BS, C, H, W = img.shape
        flow = torch.cat([flow[:, :1, :, :] / ((W-1.0)/2.0), flow[:, 1:2, :, :] / ((H-1.0)/2.0)], dim=1)
        self.grid = get_grid(BS, H, W, device=img.get_device())
        final_grid = (self.grid + flow).permute(0,2,3,1).cuda(img.get_device())
        output = self.grid_sample(img, final_grid)
        return output