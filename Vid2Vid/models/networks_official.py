import functools
import copy 

import torch.nn as nn
from .base_network import BaseNetwork

class CustomNorm(nn.Module):
    def __init__(self, norm_type, dim=None):
        super().__init__()
        self.norm_type = norm_type
        self.dim = dim
        if norm_type == "bn": self.norm_layer = nn.BatchNorm2d(dim)
        elif norm_type == "in": self.norm_layer = nn.InstanceNorm2d(dim)
        else: raise NotImplementedError(f"{norm_type} is not implemented!!!!")
    def forward(self, x):
        return self.norm_layer(x)
class ResBlk(nn.Module):
    def __init__(self, dim, padding_type, norm_type, use_dropout=False):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_type, use_dropout)
    def build_conv_block(self, dim, padding_type, norm_type, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       CustomNorm(norm_type=norm_type, dim=dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       CustomNorm(norm_type=norm_type, dim=dim)]
        return nn.Sequential(*conv_block)
    def forward(self, x):
        out = x + self.conv_block(x)
        return out
class Generator(BaseNetwork):  # composite generator in official
    def __init__(self, in_ch, output_ch, prev_output_ch, n_blks=9, ngf=128, norm_type="bn", n_down=3, padding_type="reflect", no_flow=False):
        '''
        in_ch : n_frames x ch (e.g., 3프레임씩 3채널이면 9)
        out_ch : ch (한장 예측하므로 1장에 대한 채널)
        '''
        super().__init__()
        self.no_flow = no_flow
        
        # model down A
        model_down_A = []
        model_down_A.append(nn.ReflectionPad2d(3))
        model_down_A.append(nn.Conv2d(in_ch, ngf, kernel_size=7, stride=1, padding=0))
        model_down_A.append(CustomNorm(norm_type, dim=ngf))
        model_down_A.append(nn.ReLU())
        for i in range(n_down):
            mult = 2**i
            model_down_A.append(nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1))
            model_down_A.append(CustomNorm(norm_type, dim=ngf*mult*2))
            model_down_A.append(nn.ReLU(True))
        mult = 2**n_down
        for i in range(n_blks - n_blks//2):
            model_down_A.append(ResBlk(ngf*mult, padding_type=padding_type, norm_type=norm_type))

        # model down B
        model_down_B = []
        model_down_B.append(nn.ReflectionPad2d(3))
        model_down_B.append(nn.Conv2d(prev_output_ch, ngf, kernel_size=7, stride=1, padding=0))
        model_down_B.append(CustomNorm(norm_type=norm_type, dim=ngf))
        model_down_B.append(nn.ReLU(True))
        model_down_B += copy.deepcopy(model_down_A[4:])

        # model res B
        model_res_B = []
        for i in range(n_blks//2):
            model_res_B.append(ResBlk(ngf*mult, padding_type=padding_type, norm_type=norm_type))

        # model up B
        model_up_B = []
        for i in range(n_down):
            mult = 2**(n_down-i)
            model_up_B.append(nn.ConvTranspose2d(ngf*mult, (ngf*mult)//2, kernel_size=3, stride=2, padding=1, output_padding=1))
            model_up_B.append(CustomNorm(norm_type, (ngf*mult)//2))
            model_up_B.append(nn.ReLU(True))

        # model final B
        model_final_B = []
        model_final_B.append(nn.ReflectionPad2d(3))
        model_final_B.append(nn.Conv2d(ngf, output_ch, kernel_size=7, stride=1, padding=0))
        model_final_B.append(nn.Tanh())

        # 4 flow models
        if not self.no_flow:
            model_res_flow = copy.deepcopy(model_res_B)
            model_up_flow = copy.deepcopy(model_up_B)
            model_final_flow = [
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, 2, kernel_size=7, stride=1, padding=0)
            ]
            model_final_weight = [
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, 1, kernel_size=7, stride=1, padding=0),
                nn.Sigmoid()
            ]
        self.model_down_A = nn.Sequential(*model_down_A)
        self.model_down_B = nn.Sequential(*model_down_B)
        self.model_res_B = nn.Sequential(*model_res_B)
        self.model_up_B = nn.Sequential(*model_up_B)
        self.model_final_B = nn.Sequential(*model_final_B)
        if not self.no_flow:
            self.model_res_flow = nn.Sequential(*model_res_flow)
            self.model_up_flow = nn.Sequential(*model_up_flow)
            self.model_final_flow = nn.Sequential(*model_final_flow)
            self.model_final_weight = nn.Sequential(*model_final_weight)
    
    def forward(self, real_A, prev_B):
        '''
        input_A = 도메인 A의 이미지들, 기본은 3장이다. [BS x 9 x 512 x 512]
        prev_B = 현재 B이미지를 만들기 위해 이전의 이미지 2장을 갖고온다. [BS x 6 x 512 x 512]
        '''
        downsample = self.model_down_A(real_A) + self.model_down_B(prev_B)
        img_feat = self.model_res_B(downsample)
        img_feat = self.model_up_B(img_feat)
        img = self.model_final_B(img_feat)

        flow = weight = flow_feat = None
        # flow pipeline
        if not self.no_flow:
            flow_feat = self.model_res_flow(downsample)
            flow_feat = self.model_up_flow(flow_feat)
            flow = self.model_final_flow(flow_feat)  # [BS x 2 x 512 x 512]
            weight = self.model_final_weight(flow_feat)  # [BS x 1 x 512 x 512]
        if not self.no_flow:
            img_warp = self.resample(prev_B[:, -3:], flow)
            weight_ = weight.expand_as(img)
            output_img = weight_ * img + (1-weight_) * img_warp
        else:
            output_img = img
        return output_img, flow, weight, img # output_img=img : [BS x 3 x 512 x 512], flow : [BS x 2 x 512 x 512], weight : [BS x 1 x 512 x 512]
class NLayerDiscriminator(nn.Module):  # 각 블록을 저장하고 중간 feature들을 뽑는다.
    def __init__(self, input_ch, ndf=64, n_layers=3, norm_type="bn"):
        super().__init__()
        self.n_layers = n_layers
        blks = []
        blks.append([
            nn.Conv2d(input_ch, ndf, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(0.2, True)
        ])
        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf*2, 512)
            blks.append([
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=2),
                CustomNorm(norm_type, dim=nf),
                nn.LeakyReLU(0.2, True)
            ])
        nf_prev = nf
        nf = min(nf*2, 512)
        blks.append([
            nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, padding=2),
            CustomNorm(norm_type, nf),
            nn.LeakyReLU(0.2, True)
        ])
        blks.append([
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=2)
        ])
        for n in range(len(blks)):
            setattr(self, f"blk_{n}", nn.Sequential(*blks[n]))
    def forward(self, x):
        outputs = [x]
        for n in range(self.n_layers+2):
            blk = getattr(self, f"blk_{n}")
            feat = blk(outputs[-1])
            outputs.append(feat)
        return outputs[1:]
class MultiScaleDiscriminator(BaseNetwork):
    def __init__(self, input_ch, ndf=64, n_layers=3, norm_type="bn", n_D=3):
        super().__init__()
        self.n_D = n_D
        self.n_layers = n_layers
        ndf_max = 64
        for i in range(n_D):
            D = NLayerDiscriminator(input_ch, ndf=min(ndf_max, ndf*(2**(n_D-1-i))), n_layers=n_layers, norm_type=norm_type)
            setattr(self, f"D_{i}", D)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    def forward(self, x):
        outputs = []
        for i in range(self.n_D):
            if i != 0: x = self.downsample(x)
            D = getattr(self, f"D_{i}")
            outputs.append(D(x))
        return outputs
            
            
            


                