import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class Encoder_Down(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, reflect_p, use_bias=False):
        super().__init__()
        block = [nn.ReflectionPad2d(reflect_p)]
        block.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=use_bias))
        block.append(nn.InstanceNorm2d(out_ch))
        block.append(nn.ReLU(True))
        self.block = nn.Sequential(*block)
    def forward(self, x):
        return self.block(x)    
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, reflect_p, use_bias=False):
        super().__init__()
        block = [Encoder_Down(in_ch, out_ch, k, s, p, reflect_p, use_bias=use_bias)]
        block.append(nn.ReflectionPad2d(1))
        block.append(nn.Conv2d(out_ch, out_ch, 3, 1, 0, bias=use_bias))
        block.append(nn.InstanceNorm2d(out_ch))
        self.block = nn.Sequential(*block)
    def forward(self, x):
        return x + self.block(x)        
class ResAdaLINBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=0, reflect_p=1, use_bias=False):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(reflect_p)
        self.conv1 = nn.Conv2d(in_ch, out_ch, k, s, p, bias=use_bias)
        self.norm1 = AdaLIN(out_ch)
        self.relu1 = nn.ReLU(True)
        
        self.pad2 = nn.ReflectionPad2d(reflect_p)
        self.conv2 = nn.Conv2d(out_ch, out_ch, k, s, p, bias=use_bias)
        self.norm2 = AdaLIN(out_ch)
    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        return x + out
class AdaLIN(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.rho = nn.Parameter(torch.Tensor(1, dim, 1, 1).data.fill_(0.9))
    def forward(self, x, gamma, beta):
        inst_var, inst_mean = torch.var_mean(x, dim=[2,3], keepdim=True)
        inst_out = (x - inst_mean) / torch.sqrt(inst_var + self.eps)
        layer_var, layer_mean = torch.var_mean(x, dim=[1,2,3], keepdim=True)
        layer_out = (x - layer_mean) / torch.sqrt(layer_var + self.eps)
        out = gamma.unsqueeze(2).unsqueeze(3) * (self.rho * inst_out + (1 - self.rho) * layer_out) + beta.unsqueeze(2).unsqueeze(3)
        return out
class LIN(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.rho = nn.Parameter(torch.Tensor(1, dim, 1, 1).data.fill_(0.0))
        self.gamma = nn.Parameter(torch.Tensor(1, dim, 1, 1).data.fill_(1.0))
        self.beta = nn.Parameter(torch.Tensor(1, dim, 1, 1).data.fill_(0.0))
    def forward(self, x):
        inst_var, inst_mean = torch.var_mean(x, dim=[2,3], keepdim=True)
        inst_out = (x - inst_mean) / torch.sqrt(inst_var + self.eps)
        layer_var, layer_mean = torch.var_mean(x, dim=[1,2,3], keepdim=True)
        layer_out = (x - layer_mean) / torch.sqrt(layer_var + self.eps)
        out = self.gamma * (self.rho * inst_out + (1 - self.rho) * layer_out) + self.beta
        return out
class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, use_bias=False):
        super().__init__()
        block = []
        block.append(nn.Upsample(scale_factor=2, mode="nearest"))
        block.append(nn.ReflectionPad2d(1))
        block.append(nn.Conv2d(in_ch, out_ch, k, s, p, bias=use_bias))
        block.append(LIN(out_ch))
        block.append(nn.ReLU(True))
        self.block = nn.Sequential(*block)
    def forward(self, x):
        return self.block(x)        
class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, n_down=2, ngf=64, n_blocks=6, img_size=256, light=False):
        super().__init__()
        self.light = light
        self.img_size = img_size
        encoder = []
        # 논문 Supple에 있는 그대로 구현
        # Encoder Down-sampling
        encoder.append(Encoder_Down(in_ch, ngf, 7, 1, 0, 3))
        for i in range(n_down):
            mult = 2**i
            encoder.append(Encoder_Down(ngf*mult, ngf*mult*2, 3, 2, 0, 1))
        # Encoder Bottleneck
        mult = 2 ** n_down
        for i in range(n_blocks):
            encoder.append(ResBlock(ngf*mult, ngf*mult, 3, 1, 0, 1, use_bias=False))
        self.encoder = nn.Sequential(*encoder)
        # CAM of Generator
        self.gap_fc = nn.Linear(ngf*mult, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf*mult, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf*mult*2, ngf*mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)
        # gamma, beta
        FC = []
        if self.light: # CAM할 떄 gap사용함.
            FC.append(nn.Linear(ngf*mult, ngf*mult, bias=False))
            FC.append(nn.ReLU(True))
            FC.append(nn.Linear(ngf*mult, ngf*mult, biase=False))
            FC.append(nn.ReLU(True))
        else:
            FC.append(nn.Linear((self.img_size // mult)**2 * ngf * mult, ngf*mult, bias=False))
            FC.append(nn.ReLU(True))
            FC.append(nn.Linear(ngf*mult, ngf*mult, bias=False))
            FC.append(nn.ReLU(True))
        self.FC = nn.Sequential(*FC)
        self.gamma_fc = nn.Linear(ngf*mult, ngf*mult, bias=False)  # CAM에 사용되므로 bias가 없어야한다.
        self.beta_fc = nn.Linear(ngf*mult, ngf*mult, bias=False)  # CAM에 사용되므로 bias가 없어야한다.
        # Decoder Bottleneck
        self.decoder_bottleneck = nn.ModuleList()
        for _ in range(n_blocks):
            self.decoder_bottleneck.append(ResAdaLINBlock(ngf*mult, ngf*mult, use_bias=False))
        # Decoder Up-sampling
        decoder_upsample = []
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            decoder_upsample.append(UpsampleBlock(ngf*mult, (ngf*mult)//2, k=3, s=1, p=0, use_bias=False))
        decoder_upsample.append(nn.ReflectionPad2d(3))
        decoder_upsample.append(nn.Conv2d(ngf, out_ch, kernel_size=7, stride=1, padding=0, bias=False))
        decoder_upsample.append(nn.Tanh())
        self.decoder_upsample = nn.Sequential(*decoder_upsample)
    def forward(self, x):
        BS = x.shape[0]
        x = self.encoder(x)

        # global average pooling을 통한 attention
        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.reshape(BS, -1))
        gap_weight = list(self.gap_fc.parameters())[0]  # [1 x 256]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.reshape(BS, -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]  # [1 x 256]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp] ,1)
        x = self.conv1x1(x)
        x = self.relu(x)  # [BS x 256 x 64 x 64]

        heatmap = torch.sum(x, dim=1, keepdim=True)  # [BS x 1 x 64 x 64]
        if self.light:  # 이 다음 x에서 FC를 gap로 씀
            x_ = F.adaptive_avg_pool2d(x, 1)
            x_ = self.FC(x_.reshape(BS, -1))
        else:
            x_ = self.FC(x.reshape(BS, -1))
        # gamma, beta를 위한 x를 x_로 따로 뺌.
        gamma, beta = self.gamma_fc(x_), self.beta_fc(x_)
        for dec_bottle in self.decoder_bottleneck:
            x = dec_bottle(x, gamma, beta)
        out = self.decoder_upsample(x)
        return out, cam_logit, heatmap
class DisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, reflect_p, use_bias=True):
        super().__init__()
        block = []
        block.append(nn.ReflectionPad2d(reflect_p))
        block.append(spectral_norm(nn.Conv2d(in_ch, out_ch, k, s, p, bias=use_bias)))
        block.append(nn.LeakyReLU(0.2, True))
        self.block = nn.Sequential(*block)
    def forward(self, x):
        return self.block(x)
class Discriminator(nn.Module):
    def __init__(self, in_ch, ndf=64, n_layers=5):
        super().__init__()
        # Encoder Down-sampling
        encoder = []
        encoder.append(DisBlock(in_ch, ndf, 4, 2, 0, 1, use_bias=True))
        for i in range(1, n_layers-2):
            mult = 2 ** (i-1)
            encoder.append(DisBlock(ndf*mult, ndf*mult*2, 4, 2, 0, 1, use_bias=True))
        mult = 2 ** (n_layers - 2 - 1)
        encoder.append(DisBlock(ndf*mult, ndf*mult*2, 4, 1, 0, 1, use_bias=True))  # TODO: kernel size 4 -> 3 지금 size가 안맞는다. 이 레이어 지나면 31로 됨. 아래 self.cls_conv도 마찬가지로
        self.encoder = nn.Sequential(*encoder)
        # CAM
        mult = 2 ** (n_layers - 2)
        self.gap_fc = spectral_norm(nn.Linear(ndf*mult, 1, bias=False))
        self.gmp_fc = spectral_norm(nn.Linear(ndf*mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf*mult*2, ndf*mult, 1, 1, 0, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        # Classifier
        self.pad = nn.ReflectionPad2d(1)
        self.cls_conv = spectral_norm(nn.Conv2d(ndf*mult, 1, 4, 1, 0, bias=False))
    def forward(self, x):
        BS = x.shape[0]
        x = self.encoder(x)

        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.reshape(BS, -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.reshape(BS, -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        cam_logit = torch.cat([gap_logit, gmp_logit], 1)

        x = torch.cat([gap, gmp], 1)
        x = self.conv1x1(x)
        x = self.leaky_relu(x)
    
        heatmap = torch.sum(x, dim=1, keepdim=True)  # [BS x 1 x 256/2**a x 256/2**a]
        
        x = self.pad(x)
        out = self.cls_conv(x)
        return out, cam_logit, heatmap
class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max
    def __call__(self, module):
        if hasattr(module, "rho"):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w        
if __name__ == "__main__":
    # G = Generator(3, 3)
    # D = Discriminator(3, 1)
    # sample_input = torch.randn(16,3,256,256)
    # output, cam_logit, heatmap = D(sample_input)
    # print(output.shape, cam_logit.shape, heatmap.shape)
    adalin2 = AdaLIN(64).cuda()
    gamma = torch.randn((4,1)).cuda()
    beta = torch.randn((4,1)).cuda()
    sample_input = torch.randn((4,64, 128,128)).cuda()
    output = adalin2(sample_input, gamma, beta)