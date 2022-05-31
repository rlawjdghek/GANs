import torch.nn as nn
import torch

from .base_networks import BaseNetwork
#### Generator ####
class ResBlk(nn.Module):
    def __init__(self, dim):
        super().__init__()
        layers = []
        layers.append(nn.ReflectionPad2d(1))
        layers.append(nn.Conv2d(dim, dim, 3))
        layers.append(nn.InstanceNorm2d(dim))
        layers.append(nn.ReLU())
        layers.append(nn.ReflectionPad2d(1))
        layers.append(nn.Conv2d(dim, dim, 3))
        layers.append(nn.InstanceNorm2d(dim))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x) + x
class ContentEncoder(nn.Module):
    def __init__(self, in_ch, nef=64, n_blks=3, n_downsample=2):
        super().__init__()
        layers = []
        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(in_ch, nef, 7))
        layers.append(nn.InstanceNorm2d(nef))
        layers.append(nn.ReLU())

        for i in range(n_downsample):
            mult = 2**i
            layers.append(nn.Conv2d(nef*mult, nef*mult*2, 4, 2, 1))
            layers.append(nn.InstanceNorm2d(nef*mult*2))
            layers.append(nn.ReLU())
        mult = 2**n_downsample
        for _ in range(n_blks):
            layers.append(ResBlk(nef*mult))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
class StyleEncoder(nn.Module):  # IN 없다. IN은 style을 망가뜨림.
    def __init__(self, in_ch, style_dim, nef=64, n_downsample=3):
        super().__init__()
        layers = []
        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(in_ch, nef, 7))
        layers.append(nn.ReLU())

        for i in range(2):
            mult = 2 ** i
            layers.append(nn.Conv2d(nef*mult, nef*mult*2, 4, 2, 1))
            layers.append(nn.ReLU())
        for i in range(n_downsample - 2):
            mult = 2 ** (2+i)
            layers.append(nn.Conv2d(nef*mult, nef*mult, 4, 2, 1))
            layers.append(nn.ReLU())
        
        mult = 2 ** n_downsample
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Conv2d(nef*mult, style_dim, 1, 1, 0))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
class Encoder(BaseNetwork):
    def __init__(self, in_ch, nef=64, n_blks=3, n_downsample=2, style_dim=8):
        super().__init__()
        self.content_encoder = ContentEncoder(in_ch, nef=nef, n_blks=n_blks, n_downsample=n_downsample)
        self.style_encoder = StyleEncoder(in_ch, style_dim=style_dim, nef=nef, n_downsample=n_downsample)
    def forward(self, x):
        content = self.content_encoder(x)
        style = self.style_encoder(x)
        return content, style  # style : [BS x style_dim x 1 x 1]        
class AdaIN(nn.Module):
    def __init__(self, style_dim, dim, out_dim, n_mlp=3):
        super().__init__()
        self.norm_layer = nn.InstanceNorm2d(out_dim)
        mean_mlp = []
        mlp = []
        mlp.append(nn.Linear(style_dim, dim))
        mlp.append(nn.ReLU())
        for _ in range(n_mlp - 2):
            mlp.append(nn.Linear(dim,dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(dim, out_dim*2))
        self.mlp = nn.Sequential(*mlp)
    def forward(self, x, style):
        style = self.mlp(style.flatten(1))
        BS, N = style.shape
        beta = style[:, :N//2].unsqueeze(-1).unsqueeze(-1)
        gamma = style[:, N//2:].unsqueeze(-1).unsqueeze(-1)
        x = self.norm_layer(x)
        return x * gamma + beta     
class AdaInResBlk(nn.Module):
    def __init__(self, dim, style_dim):
        super().__init__()
        blk_1 = []
        blk_1.append(nn.ReflectionPad2d(1))
        blk_1.append(nn.Conv2d(dim, dim, 3))
        self.blk_1 = nn.Sequential(*blk_1)
        self.adain_1 = AdaIN(style_dim, dim, dim)
        self.act_1 = nn.ReLU()
        blk_2 = []
        blk_2.append(nn.ReflectionPad2d(1))
        blk_2.append(nn.Conv2d(dim, dim, 3))
        self.blk_2 = nn.Sequential(*blk_2)
        self.adain_2 = AdaIN(style_dim, dim, dim)
    def forward(self, x, style):
        x_skip = x
        x = self.blk_1(x)
        x = self.adain_1(x, style)
        x = self.act_1(x)
        x = self.blk_2(x)
        x = self.adain_2(x, style)
        return x + x_skip
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
class Decoder(BaseNetwork):
    def __init__(self, out_ch, ngf=64, style_dim=8, n_blks=3, n_upsample=2):
        super().__init__()
        self.res_blks = nn.ModuleList()
        for _ in range(n_blks):
            self.res_blks.append(AdaInResBlk(ngf * (2**n_upsample), style_dim))
        upsample = []
        for i in range(n_upsample):
            mult = 2 ** (n_upsample - i)
            upsample.append(nn.Upsample(scale_factor=2))
            upsample.append(nn.Conv2d(ngf*mult, ngf*mult//2, kernel_size=5, stride=1, padding=2))
            upsample.append(LayerNorm(ngf*mult//2))
            upsample.append(nn.ReLU())
        upsample.append(nn.ReflectionPad2d(3))
        upsample.append(nn.Conv2d(ngf, out_ch, 7))
        upsample.append(nn.Tanh())
        self.upsample = nn.Sequential(*upsample)
    def forward(self, content, style):
        x = content
        for res_blk in self.res_blks:
            x = res_blk(x, style)
        output = self.upsample(x)
        return output
#### Discriminator ####
class CustomNorm(nn.Module):
    def __init__(self, norm_type, dim):
        super().__init__()
        self.norm_type = norm_type
        self.dim = dim
        if norm_type == "bn": self.norm_layer = nn.BatchNorm2d(dim)
        elif norm_type == "in": self.norm_layer = nn.InstanceNorm2d(dim)
    def forward(self, x):
        return self.norm_layer(x)
class NLayerDiscriminator(BaseNetwork):
    def __init__(self, in_ch, ndf=64, n_layers=3, norm_type="in"):
        super().__init__()
        self.n_layers = n_layers
        blks = []
        blks.append([
            nn.Conv2d(in_ch, ndf, 4, 2, 2),
            nn.LeakyReLU(0.2, True)
        ])
        nf = ndf
        for _ in range(1, n_layers):
            nf_prev = nf
            nf = min(nf*2, 512)
            blks.append([
                nn.Conv2d(nf_prev, nf, 4, 2, 2),
                CustomNorm(norm_type, nf),
                nn.LeakyReLU(0.2, True)
            ])
        nf_prev = nf
        nf = min(nf*2, 512)
        blks.append([
            nn.Conv2d(nf_prev, nf, 4, 1, 2),
            CustomNorm(norm_type, nf),
            nn.LeakyReLU(0.2, True)
        ])
        blks.append([nn.Conv2d(nf, 1, 4, 1, 2)])
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
    def __init__(self, in_ch, ndf=64, n_layers=3, norm_type="in", n_D=3):
        super().__init__()
        self.n_D = n_D
        self.n_layers = n_layers
        ndf_max = 64
        for i in range(n_D):
            D = NLayerDiscriminator(in_ch, ndf=min(ndf_max, ndf*(2**(n_D-i))), n_layers=n_layers, norm_type=norm_type)
            setattr(self, f"D_{i}", D)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    def forward(self, x):
        outputs = []
        for i in range(self.n_D):
            if i != 0: x = self.downsample(x)
            D = getattr(self, f"D_{i}")
            outputs.append(D(x))
        return outputs
