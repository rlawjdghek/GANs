import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
def define_network_G(args, input_ch, output_ch, prev_output_ch):
    if args.G_name == "official":
        from .networks_official import Generator
        G = Generator(input_ch, output_ch, prev_output_ch)
    return G
def define_network_D(args, input_ch):
    if args.D_name == "multi_scale":
        from .networks_official import MultiScaleDiscriminator
        D = MultiScaleDiscriminator(input_ch, ndf=args.ndf, n_layers=args.n_layers_D, norm_type=args.norm_type_D, n_D=args.n_D)
    return D
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
class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
    def forward(self, x, label, mask):
        mask = mask.expand(-1, x.shape[1], -1, -1)
        loss = self.criterion(x*mask, label*mask)
        return loss
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
class VGGLoss(nn.Module):
    def __init__(self, local_rank):
        super().__init__()
        self.vgg = Vgg19().cuda(local_rank)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]  # 해상도별 feature별로 ch는 2배 늘고 해상도는 2배 줄어드므로 픽셀 갯수는 2배씩 줄어든다.
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
    def forward(self, x, y):
        while x.shape[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        loss = 0
        for i in range(len(x_features)):
            loss += self.weights[i] * self.criterion(x_features[i], y_features[i].detach())
        return loss           
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
class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    def grid_sample(self, input1, input2):
        return F.grid_sample(input1, input2, mode="bilinear", padding_mode="border")
    def resample(self, img, flow):
        BS, C, H, W = img.shape
        flow = torch.cat([flow[:, :1, :, :] / ((W-1.0)/2.0), flow[:, 1:2, :, :] / ((H-1.0)/2.0)], dim=1)
        self.grid = get_grid(BS, H, W, device=img.get_device())
        final_grid = (self.grid + flow).permute(0,2,3,1).cuda(img.get_device())
        output = self.grid_sample(img, final_grid)
        return output
