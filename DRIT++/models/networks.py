from copy import deepcopy

import torch
import torch.nn as nn
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .base_network import BaseNetwork

class LeakyReLUConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding=0, norm="None"):
        super().__init__()
        blk = []
        blk.append(nn.ReflectionPad2d(padding))
        blk.append(spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, bias=True)))
        if norm == "in":
            blk.append(nn.InstanceNorm2d(out_ch))
        blk.append(nn.LeakyReLU(True))
        self.blk = nn.Sequential(*blk)
    def forward(self, x):
        return self.blk(x)
class ReLUINConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding=0):
        super().__init__()
        blk = []
        blk.append(nn.ReflectionPad2d(padding))
        blk.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, bias=True))
        blk.append(nn.InstanceNorm2d(out_ch, affine=False))
        blk.append(nn.ReLU(True))
        self.blk = nn.Sequential(*blk)
    def forward(self, x):
        return self.blk(x)
def Conv3x3(in_ch, out_ch, stride=1):
    return [nn.ReflectionPad2d(1), nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1)]
# class Conv3x3(nn.Module):
#     def __init__(self, in_ch, out_ch, stride=1):
#         super().__init__()
#         blk = []
#         blk.append(nn.ReflectionPad2d(1))
#         blk.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride))
#         self.blk = nn.Sequential(*blk)
#     def forward(self, x):
#         return self.blk(x)
class INResBlk(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        blk = []
        blk.extend(Conv3x3(in_ch, out_ch, stride))
        blk.append(nn.InstanceNorm2d(out_ch))
        blk.append(nn.ReLU(True))
        blk.extend(Conv3x3(out_ch, out_ch))
        blk.append(nn.InstanceNorm2d(out_ch))
        if dropout > 0:
            blk.append(nn.Dropout(p=dropout))
        self.blk = nn.Sequential(*blk)
    def forward(self, x):
        res = x
        out = self.blk(x)
        out = out + res
        return out
class GauNoiseBlk(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if self.training == False:
            return x
        noise = torch.randn(x.shape).cuda(x.get_device())
        return x + noise
class MisINResBlk(nn.Module):
    def __init__(self, dim, dim_extra, stride=1):
        super().__init__()
        conv1 = []
        conv1.extend(Conv3x3(dim, dim, stride=stride))
        conv1.append(nn.InstanceNorm2d(dim))
        conv2 = deepcopy(conv1)
        blk1 = []
        blk1.append(nn.Conv2d(dim+dim_extra, dim+dim_extra, 1, 1, 0))
        blk1.append(nn.ReLU(True))
        blk1.append(nn.Conv2d(dim+dim_extra, dim, 1, 1, 0))
        blk1.append(nn.ReLU(True))
        blk2 = deepcopy(blk1)
        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Sequential(*conv2)
        self.blk1 = nn.Sequential(*blk1)
        self.blk2 = nn.Sequential(*blk2)
    def forward(self, x, z):
        res = x
        z_expand = z.reshape(z.shape[0], z.shape[1], 1, 1).expand(z.shape[0], z.shape[1], x.shape[2], x.shape[3])
        o1 = self.conv1(x)
        o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
        o3 = self.conv2(o2)
        out = self.blk2(torch.cat([o3, z_expand], dim=1))
        out = out + res
        return out
class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(x, normalized_shape)
class ReLULNConvT2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, output_padding):
        super().__init__()
        blk = []
        blk.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True))
        blk.append(LayerNorm(out_ch))
        blk.append(nn.ReLU(True))
        self.blk = nn.Sequential(*blk)
    def forward(self, x):
        out = self.blk(x)
        return out
class EncoderContent(BaseNetwork):
    def __init__(self, in_ch_A, in_ch_B, nef=64):
        super().__init__()
        tmp_nef = nef
        convA = []
        convA.append(LeakyReLUConv2d(in_ch_A, tmp_nef, kernel_size=7, stride=1, padding=3))
        for _ in range(2):
            convA.append(ReLUINConv2d(tmp_nef, tmp_nef*2, kernel_size=3, stride=2, padding=1))
            tmp_nef *= 2
        for _ in range(3):
            convA.append(INResBlk(tmp_nef, tmp_nef))
        
        tmp_nef = nef
        convB = []
        convB.append(LeakyReLUConv2d(in_ch_B, tmp_nef, kernel_size=7, stride=1, padding=3))
        for _ in range(2):
            convB.append(ReLUINConv2d(tmp_nef, tmp_nef*2, kernel_size=3, stride=2, padding=1))
            tmp_nef *= 2
        for _ in range(3):
            convB.append(INResBlk(tmp_nef, tmp_nef))
        
        conv_share = []
        conv_share.append(INResBlk(tmp_nef, tmp_nef))
        conv_share.append(GauNoiseBlk())
               
        self.convA = nn.Sequential(*convA)
        self.convB = nn.Sequential(*convB)
        self.conv_share = nn.Sequential(*conv_share)
    def forward(self, A, B):
        outputA = self.convA(A)
        outputB = self.convB(B)
        outputA = self.conv_share(outputA)
        outputB = self.conv_share(outputB)
        return outputA, outputB
class EncoderStyle(BaseNetwork):
    def __init__(self, in_ch_A, in_ch_B, style_dim=8, nef=64):
        super().__init__()
        convA = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch_A, nef, 7, 1),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(nef, nef*2, 4, 2),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(nef*2, nef*4, 4, 2),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(nef*4, nef*4, 4, 2),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(nef*4, nef*4, 4, 2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nef*4, style_dim, 1, 1, 0)
        ]
        convB = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch_B, nef, 7, 1),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(nef, nef*2, 4, 2),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(nef*2, nef*4, 4, 2),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(nef*4, nef*4, 4, 2),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(nef*4, nef*4, 4, 2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nef*4, style_dim, 1, 1, 0),
        ]
        self.convA = nn.Sequential(*convA)
        self.convB = nn.Sequential(*convB)
    def forward_A(self, A):
        A = self.convA(A)
        output_A = A.reshape(A.shape[0], -1)
        return output_A
    def forward_B(self, B):
        B = self.convB(B)
        output_B = B.reshape(B.shape[0], -1)
        return output_B
    def forward(self, A, B):
        A = self.forward_A(A)
        B = self.forward_B(B)
        return A, B
class Generator(BaseNetwork):
    def __init__(self, out_ch_A, out_ch_B, ngf=256, style_dim=8):
        super().__init__()
        self.ngf = ngf
        self.mlpA = nn.Sequential(
            nn.Linear(style_dim, ngf),
            nn.ReLU(True),
            nn.Linear(ngf,ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf*4)
        )
        self.mlpB = deepcopy(self.mlpA)

        self.decA1 = MisINResBlk(ngf, ngf)
        self.decA2 = MisINResBlk(ngf, ngf)
        self.decA3 = MisINResBlk(ngf, ngf)
        self.decA4 = MisINResBlk(ngf, ngf)
        tmp_ngf = ngf
        decA5 = []  
        decA5.append(ReLULNConvT2d(tmp_ngf, tmp_ngf//2, kernel_size=3, stride=2, padding=1, output_padding=1))
        tmp_ngf = tmp_ngf//2
        decA5.append(ReLULNConvT2d(tmp_ngf, tmp_ngf//2, kernel_size=3, stride=2, padding=1, output_padding=1))
        tmp_ngf = tmp_ngf//2
        decA5.append(nn.ConvTranspose2d(tmp_ngf, out_ch_A, kernel_size=1, stride=1, padding=0))
        decA5.append(nn.Tanh())
        self.decA5 = nn.Sequential(*decA5)

        self.decB1 = MisINResBlk(ngf, ngf)
        self.decB2 = MisINResBlk(ngf, ngf)
        self.decB3 = MisINResBlk(ngf, ngf)
        self.decB4 = MisINResBlk(ngf, ngf)
        tmp_ngf = ngf
        decB5 = []  
        decB5.append(ReLULNConvT2d(tmp_ngf, tmp_ngf//2, kernel_size=3, stride=2, padding=1, output_padding=1))
        tmp_ngf = tmp_ngf//2
        decB5.append(ReLULNConvT2d(tmp_ngf, tmp_ngf//2, kernel_size=3, stride=2, padding=1, output_padding=1))
        tmp_ngf = tmp_ngf//2
        decB5.append(nn.ConvTranspose2d(tmp_ngf, out_ch_B, kernel_size=1, stride=1, padding=0))
        decB5.append(nn.Tanh())
        self.decB5 = nn.Sequential(*decB5)
    def forward_A(self, content, style):
        '''
        content : real_B, real_A, real_B, real_B로부터 생성된 content
        style : real_A, real_A, random_A1, random_A2로부터 생성된 style
        '''
        style = self.mlpA(style)  # [4BS x 1024]
        style1, style2, style3, style4 = torch.split(style, self.ngf, dim=1)  # [4BS x 1024]
        style1, style2, style3, style4 = style1.contiguous(), style2.contiguous(), style3.contiguous(), style4.contiguous()
        out1 = self.decA1(content, style1)
        out2 = self.decA2(out1, style2)
        out3 = self.decA3(out2, style3)
        out4 = self.decA4(out3, style4)
        out = self.decA5(out4)
        return out
    def forward_B(self, content, style):
        style = self.mlpB(style)  # [4BS x 1024]
        style1, style2, style3, style4 = torch.split(style, self.ngf, dim=1)  # [4BS x 1024]
        style1, style2, style3, style4 = style1.contiguous(), style2.contiguous(), style3.contiguous(), style4.contiguous()
        out1 = self.decB1(content, style1)
        out2 = self.decB2(out1, style2)
        out3 = self.decB3(out2, style3)
        out4 = self.decB4(out3, style4)
        out = self.decB5(out4)
        return out
#######################
#### Discriminator ####
#######################
class DiscriminatorContent(BaseNetwork):
    def __init__(self, ndf=256):
        super().__init__()
        blk = []
        blk.append(LeakyReLUConv2d(ndf, ndf, kernel_size=7, stride=2, padding=1, norm="in"))
        blk.append(LeakyReLUConv2d(ndf, ndf, kernel_size=7, stride=2, padding=1, norm="in"))
        blk.append(LeakyReLUConv2d(ndf, ndf, kernel_size=7, stride=2, padding=1, norm="in"))
        blk.append(LeakyReLUConv2d(ndf, ndf, kernel_size=4, stride=1, padding=0))
        blk.append(nn.Conv2d(ndf, 1, 1, 1, 0))
        self.blk = nn.Sequential(*blk)
    def forward(self, x):
        x = self.blk(x)
        return [x.reshape(-1)]
class NLayerDiscriminator(BaseNetwork):
    def __init__(self, in_ch, ndf, n_layer, norm="None"):
        super().__init__()
        blk = []
        blk.append(LeakyReLUConv2d(in_ch, ndf, kernel_size=3,stride=2, padding=1, norm=norm))
        for _ in range(1, n_layer - 1):
            blk.append(LeakyReLUConv2d(ndf, ndf*2, kernel_size=3, stride=2, padding=1, norm=norm))
            ndf *= 2
        blk.append(LeakyReLUConv2d(ndf, ndf*2, kernel_size=3, stride=2, padding=1, norm="None"))
        ndf *= 2
        blk.append(spectral_norm(nn.Conv2d(ndf, 1, kernel_size=1, stride=1, padding=0)))
        self.blk = nn.Sequential(*blk)
    def forward(self, x):
        x = self.blk(x)
        return [x.reshape(-1)]
class ScaleDiscriminator(BaseNetwork):
    def __init__(self, in_ch, ndf, n_layer, norm):
        super().__init__()
        blk = []
        blk.append(LeakyReLUConv2d(in_ch, ndf, 4, 2, 1, norm))
        for i in range(1, n_layer):
            blk.append(LeakyReLUConv2d(ndf, ndf*2, 4, 2, 1, norm))
            ndf *= 2
        blk.append(spectral_norm(nn.Conv2d(ndf, 1, 1, 1, 0)))
        self.blk = nn.Sequential(*blk)
    def forward(self, x):
        return self.blk(x)
class MultiScaleDiscriminator(BaseNetwork):
    def __init__(self, in_ch, ndf, n_D, n_layer, norm="None"):
        super().__init__()
        self.n_D = n_D
        ndf_max = 64
        for i in range(n_D):
            D = ScaleDiscriminator(in_ch, ndf=min(ndf_max, ndf*(2**(n_D - i))), n_layer=n_layer, norm=norm)
            setattr(self, f"D_{i}", D)
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
    def forward(self, x):
        outputs = []
        for i in range(self.n_D):
            D = getattr(self, f"D_{i}")
            output = D(x)
            outputs.append(output)
            x = self.downsample(x)
        return outputs
        
        

    
        

        
        

        
        
        
        

        


