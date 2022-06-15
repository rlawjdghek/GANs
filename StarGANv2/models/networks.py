import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_network import BaseNetwork

class CustomNorm(nn.Module):
    def __init__(self, dim, norm_type):
        super().__init__()
        if norm_type == "in":
            self.norm_layer = nn.InstanceNorm2d(dim, affine=True)
        elif norm_type == "none":
            self.norm_layer = nn.Identity()
    def forward(self, x):
        return self.norm_layer(x)
class CustomAct(nn.Module):
    def __init__(self, act_type):
        super().__init__()
        if act_type == "leaky":
            self.act_layer = nn.LeakyReLU(0.2)
        elif act_type == "relu":
            self.act_layer = nn.ReLU()
    def forward(self, x):
        return self.act_layer(x)
class ResBlk(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type="in", act_type="leaky", downsample=False, normalize=False):
        super().__init__()
        self.downsample = downsample
        self.normalize = normalize
        self.act_layer = CustomAct(act_type)
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        if self.normalize:
            self.norm1 = CustomNorm(in_ch, norm_type)
            self.norm2 = CustomNorm(in_ch, norm_type)
        self.shortcut = in_ch != out_ch
        if self.shortcut:
            self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)           
    def _shortcut(self, x):
        if self.shortcut:
            x = self.conv_shortcut(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x
    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.act_layer(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        return x
    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)
class AdaIN(nn.Module):
    def __init__(self, dim, style_dim):
        super().__init__()
        self.norm_layer = nn.InstanceNorm2d(dim, affine=False)
        self.fc = nn.Linear(style_dim, dim*2)
    def forward(self, x, style):
        style = self.fc(style).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(style, chunks=2, dim=1)
        return (1+gamma) * self.norm_layer(x) + beta
class AdaINResBlk(nn.Module):
    def __init__(self, in_ch, out_ch, style_dim=64, w_hpf=0, upsample=False, act_type="leaky"):
        super().__init__()
        self.w_hpf = w_hpf
        self.upsample = upsample
        self.shortcut = in_ch != out_ch
        self.conv1 = nn.Conv2d(in_ch, out_ch ,3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm1 = AdaIN(in_ch, style_dim)
        self.norm2 = AdaIN(out_ch, style_dim)
        self.act_layer = CustomAct(act_type)
        if self.shortcut:
            self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.shortcut:
            x = self.conv_shortcut(x)
        return x
    def _residual(self, x, style):
        x = self.norm1(x, style)
        x = self.act_layer(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.norm2(x, style)
        x = self.act_layer(x)
        x = self.conv2(x)
        return x
    def forward(self, x, style):
        out = self._residual(x, style)
        if self.w_hpf == 0:  # 하이패스 필터가 적용되면 skip connetion 안함.
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out
class Highpass(nn.Module):
    def __init__(self, w_hpf):
        super().__init__()
        self.register_buffer(
            "filter", torch.tensor([[-1,-1,-1],
                                    [-1,8.0,-1], 
                                    [-1,-1,-1]]) / w_hpf)
    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.shape[1])  # depthwise convolution
class Generator(BaseNetwork):
    def __init__(self, in_ch, out_ch, img_size=256, style_dim=64, max_ngf=512, w_hpf=0):
        super().__init__()
        self.img_size = img_size
        ngf = (2**14) // img_size  # 64
        self.from_rgb = nn.Conv2d(in_ch, ngf, 3, 1, 1)
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(ngf, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf, out_ch, 1, 1, 0)
        )
        n_downsample = int(math.log2(img_size)) - 4  # 최소 해상도는 16x16

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        #### downsample & upsample ####
        prev_ngf = ngf
        for i in range(n_downsample):
            ngf = min(prev_ngf*2, max_ngf)
            self.encoder.append(ResBlk(prev_ngf, ngf, norm_type="in", downsample=True, normalize=True))
            self.decoder.insert(0, AdaINResBlk(ngf, prev_ngf, style_dim=style_dim, w_hpf=w_hpf, upsample=True))
            prev_ngf = ngf

        #### bottleneck ####
        for i in range(2):
            self.encoder.append(ResBlk(ngf, ngf, norm_type="in", downsample=False, normalize=True))
            self.decoder.insert(0, AdaINResBlk(ngf, ngf, style_dim=style_dim, w_hpf=w_hpf, upsample=False))
        
        if w_hpf > 0:
            self.hpf = Highpass(w_hpf)
    def forward(self, x, style, masks=None):
        x = self.from_rgb(x)
        cache = {}
        for blk in self.encoder:
            if (masks is not None) and (x.shape[2] in [32,64,128]):
                cache[x.shape[2]] = x
            x = blk(x)
        for blk in self.decoder:
            x = blk(x, style)
            if (masks is not None) and (x.shape[2] in [32,64,128]):
                mask = masks[0] if x.shape[2] in [32] else masks[1]
                mask = F.interpolate(mask, size=x.shape[2], model="bilinear")
                x = x + self.hpf(mask * cache[x.shape[2]])
        return self.to_rgb(x)
class Discriminator(BaseNetwork):
    def __init__(self, in_ch, img_size=256, n_domains=2, max_ndf=512):
        super().__init__()
        ndf = 2**14 // img_size  # 64
        n_downsample = int(math.log2(img_size)) - 2  # 무조건 해상도를 4로 맞춤
        blk = []
        blk.append(nn.Conv2d(in_ch, ndf, 3, 1, 1))
        prev_ndf = ndf
        for _ in range(n_downsample):
            ndf = min(prev_ndf*2, max_ndf)
            blk.append(ResBlk(prev_ndf, ndf, downsample=True))
            prev_ndf = ndf
        blk.append(nn.LeakyReLU(0.2))
        blk.append(nn.Conv2d(ndf, ndf, 4, 1, 0))  # [BS x ndf x 1 x 1]
        blk.append(nn.LeakyReLU(0.2))
        self.blk = nn.Sequential(*blk)
        self.fc = nn.Linear(ndf, n_domains)
    def forward(self, x, y):
        BS = x.shape[0]
        out = self.blk(x)  # [BS x ndf x 1 x 1]
        out = out.flatten(1)
        out = self.fc(out)  # [BS x n_domain]
        idx = torch.arange(BS).to(y.device)
        out = out[idx, y]
        return out
class MappingNetwork(BaseNetwork):
    def __init__(self, latent_dim=16, style_dim=64, n_domains=2):
        super().__init__()
        shared_layer = []
        shared_layer.append(nn.Linear(latent_dim, 512))
        shared_layer.append(nn.ReLU())
        for _ in range(3):
            shared_layer.append(nn.Linear(512, 512))
            shared_layer.append(nn.ReLU())
        self.shared_layer = nn.Sequential(*shared_layer)
        
        self.unshared_layers = nn.ModuleList()
        for _ in range(n_domains):
            unshared_layer = nn.Sequential(
                nn.Linear(512,512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, style_dim)
            )
            self.unshared_layers.append(unshared_layer)
    def forward(self, z, y):
        style = self.shared_layer(z)
        styles = []
        for layer in self.unshared_layers:
            styles.append(layer(style))
        styles = torch.stack(styles, dim=1)  # [BS x n_domains x style_dim]
        BS = styles.shape[0]
        idx = torch.arange(BS).to(y.device)
        return styles[idx, y]
class StyleEncoder(BaseNetwork):
    def __init__(self, in_ch, img_size=256, style_dim=64, n_domains=2, max_nef=512):
        super().__init__()
        nef = 2**14 // img_size
        n_downsample = int(math.log2(img_size)) - 2  # 무조건 해상도를 4로 맞춤
        blk = []
        blk.append(nn.Conv2d(in_ch, nef, 3, 1, 1))
        prev_nef = nef
        for _ in range(n_downsample):
            nef = min(prev_nef*2, max_nef)
            blk.append(ResBlk(prev_nef, nef, downsample=True))
            prev_nef = nef
        blk.append(nn.LeakyReLU(0.2))
        blk.append(nn.Conv2d(nef, nef, 4, 1, 0))  # [BS x ndf x 1 x 1]
        blk.append(nn.LeakyReLU(0.2))
        self.blk = nn.Sequential(*blk)

        self.unshared_layers = nn.ModuleList()
        for _ in range(n_domains):
            self.unshared_layers.append(nn.Linear(nef, style_dim))
    def forward(self, x, y):
        x = self.blk(x)
        x = x.flatten(1)
        styles = []
        for layer in self.unshared_layers:
            styles.append(layer(x))
        styles = torch.stack(styles, dim=1)  # [BS x n_domains x style_dim]
        BS = styles.shape[0]
        idx = torch.arange(BS).to(y.device)
        return styles[idx, y]
        
        



            
        
        




            
            

        
        
        

    
