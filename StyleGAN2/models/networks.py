import math
import random
from cv2 import blur 

import torch
import torch.nn as nn
import torch.nn.functional as F

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix

class PixelNorm(nn.Module):  # PGGAN에서 나온 Pixelwise Feature 정규화
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
class EqualLinear(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
        bias_init=0,
        lr_mul=1,
        activation=None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias: self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else: self.bias = None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul  
    def forward(self, x):
        if self.activation:
            x = F.linear(x, self.weight * self.scale)
            out = fused_leaky_relu(x, self.bias * self.lr_mul)
        else: out = F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out
class EqualConv2d(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride=1,
        padding=0,
        bias=True
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_ch * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        if bias: self.bias = nn.Parameter(torch.zeros(out_ch))
        else: self.bias = None
    def forward(self, x):
        out = conv2d_gradfix.conv2d(
            input=x,
            weight=self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding           
        )
        return out
class ConstantInput(nn.Module):  # 입력으로 constant
    def __init__(self, n_channels, size=4):
        super().__init__()
        self.cinput = nn.Parameter(torch.zeros(1, n_channels, size, size))
    def forward(self, x):
        bs = x.shape[0]
        return self.cinput.repeat(bs, 1, 1, 1)
def make_kernel(k):  # Blur 커널 만뜰떄 [1,3,3,1] 같은 리스트를 받아서 2차원 blur kernel을 만든다. 
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim==1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k
class Blur(nn.Module):  # 말 그래도 블러링 커널. 학습이 되지 않도록 해준다. 
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        self.pad = pad
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer("kernel", kernel)
    def forward(self, x):
        out = upfirdn2d(x, self.kernel, pad=self.pad)
        return out
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
    def forward(self, x, noise=None):
        if noise is None:
            bs, _, h, w = x.shape
            noise = x.new_empty((bs, 1, h, w)).normal_()
        return x + self.weight * noise
class Upsample(nn.Module):  # factor 배수로 늘어난다.
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)
        p = kernel.shape[0] - factor
        pad0 = (p+1) // 2 + factor - 1
        pad1 = p // 2 
        self.pad = (pad0, pad1)
    def forward(self, x):
        return upfirdn2d(x, self.kernel, up=self.factor, down=1, pad=self.pad)        
class ModulatedConv2d(nn.Module):
    def __init__(
        self, 
        in_ch,
        out_ch,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1,3,3,1],
        fused=True
    ):
        super().__init__()
        self.eps = 1e-8
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.downsample = downsample
        self.fused = fused
        
        if self.upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p+1) // 2  + factor - 1
            pad1 = p // 2 + 1  # kernel_size=3이고 나머지는 default라면 pad0=1, pad1=1
            self.blur = Blur(blur_kernel, pad=[pad0, pad1], upsample_factor=factor)
        if self.downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p+1) // 2
            pad1 = p // 2 + 1  # kernel_size=3이고 나머지는 default라면 pad0=0, pad1=1
            self.blur = Blur(blur_kernel, pad=[pad0, pad1])
        fan_in = in_ch * kernel_size ** 2  # input neuron의 갯수.
        self.scale = 1 / math.sqrt(fan_in)  # Eq. (1)에서 s
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel_size, kernel_size))
        self.modulation = EqualLinear(style_dim, in_ch, bias_init=1)
    def forward(self, x, style):  # 논문에서 그림 d를 보면 된다. x는 이전 레이어의 output, style은 A
        bs, in_ch, h, w = x.shape
        if not self.fused:  # 여기서는 x에 style을 곱함
            style = self.modulation(style)  # [bs x in_ch]
            weight = self.scale * self.weight.squeeze(0) 
            if self.demodulate:  # 논문에서 Eq. (3)의 분모를 구함. 이 때 style까지 곱해진 weight의 sum을 구함.
                w = weight.unsqueeze(0) * style.reshape(bs, 1, in_ch, 1, 1) # w : [bs x out_ch x in_ch x k x k]
                dcoefs = (w.square().sum((2,3,4)) + 1e-8).rsqrt()  # 마지막에 루트 씌우고 역수변환, [bs x out_ch]
            x = x * style.reshape(bs, in_ch, 1, 1)  # x : [bs x in_ch x h x w], 스타일 입히기. 
            # input이 i.i.d이라 가정했으므로 스타일을 입힐 때 그냥 스타일을 곱해준다. 
            if self.upsample:
                weight = weight.transpose(0,1)  # w : [out_ch x bs x in_ch x k x k]
                out = conv2d_gradfix.conv_transpose2d(x, weight, padding=0, stride=2)
                out = self.blur(out)
            elif self.downsample:
                x = self.blur(x)
                out = conv2d_gradfix.conv2d(x, weight, padding=0, stride=2)
            else: out = conv2d_gradfix.conv2d(x, weight, padding=self.padding)
            # 만약 demodulate 켜져있으면 위에서 구한 dcoef로 Eq. (3) 연산.
            if self.demodulate:
                out = out * dcoefs.reshape(bs, -1, 1, 1)
            return out
        else:  # 여기서는 style을 weight에 곱한다. 논문 그림의 (d)
            style = self.modulation(style).reshape(bs, 1, -1, 1, 1)
            weight = self.scale * self.weight * style  # 스타일을 weight에 입힘. 논문의 Eq. (1)
            
            if self.demodulate:
                dcoefs = (weight.square().sum((2,3,4)) + 1e-8).rsqrt()
                weight = weight * dcoefs.reshape(bs, -1, 1, 1, 1) # weight : [bs x out_ch x in_ch x k x k] 논문의 Eq. (3)
            weight = weight.reshape(bs * self.out_ch, in_ch, self.kernel_size, self.kernel_size)
            if self.upsample:
                x = x.reshape(1, bs * in_ch, h, w)
                weight = weight.reshape(bs * self.out_ch, in_ch, self.kernel_size, self.kernel_size)
                weight = weight.transpose(1, 2).reshape(
                    bs * in_ch, self.out_ch, self.kernel_size, self.kernel_size
                )
                out = conv2d_gradfix.conv_transpose2d(x, weight, padding=0, stride=2, groups=bs)
                _, _, h, w = out.shape
                out = out.reshape(bs, self.out_ch, h, w)
                out = self.blur(out)
            elif self.downsample:
                x = self.blur(x)
                _, _, h, w = x.shape
                x = x.reshape(1, bs * in_ch, h, w)
                out = conv2d_gradfix.conv2d(x, weight, padding=0, stride=2, groups=bs)
                _, _, h, w = out.shape
                out = out.reshape(bs, self.out_ch, h, w)
            else:
                x = x.reshape(1, bs * in_ch, h, w)
                out = conv2d_gradfix.conv2d(x, weight, padding=self.padding, groups=bs)
                _, _, h, w = out.shape
                out = out.reshape(bs, self.out_ch, h, w)
            return out
class ToRGB(nn.Module):  # 마지막 conv. out_ch = 3
    def __init__(self, in_ch, style_dim, upsample=True, blur_kernel=[1,3,3,1]):
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = ModulatedConv2d(in_ch, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1,3,1,1))
    def forward(self, x, style, skip=None):
        x = self.conv(x, style)
        out = x + self.bias
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out
class StyleConv(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1,3,3,1],
        demodulate=True,
    ):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_ch=in_ch,
            out_ch=out_ch,
            kernel_size=kernel_size,
            style_dim=style_dim,
            demodulate=demodulate,
            upsample=upsample,
            blur_kernel=blur_kernel
        )
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_ch)
    def forward(self, x, style, noise=None):
        x = self.conv(x, style)
        x = self.noise(x, noise=noise)
        return self.activate(x)        
class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1,3,3,1],
        lr_mlp=0.01
    ):
        super().__init__()
        self.style_dim = style_dim
        self.channels = {
            4: 512,
            8: 512, 
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier
        }
        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1  # 1024일 경우 17. 
        self.cinput = ConstantInput(self.channels[4], size=4)
        self.n_styles = self.num_layers + 1  # 1024일 경우 18
        # mapping network
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(self.style_dim, self.style_dim, lr_mul=lr_mlp, activation="fused_lrelu"))
        self.style = nn.Sequential(*layers)

        # generator
        self.conv1 = StyleConv(
            in_ch=self.channels[4],
            out_ch=self.channels[4],
            kernel_size=3,
            style_dim=style_dim,
            blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)
        self.convs  =nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        in_ch = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn((1,1,2**res,2**res)))
            # 노이즈는 이 고정된 것을 쓸 수도 있고 아무 노이즈나 쓸 수도 있다. 
        # 중간중간 해상도마다 스타일 레이어 추가
        for i in range(3, self.log_size + 1):
            out_ch = self.channels[2 ** i]
            self.convs.append(
                StyleConv(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=3,
                    style_dim=style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel
                )
            )
            self.convs.append(
                StyleConv(
                    in_ch=out_ch,
                    out_ch=out_ch,
                    kernel_size=3,
                    style_dim=style_dim,
                    blur_kernel=blur_kernel
                )
            )
            self.to_rgbs.append(
                ToRGB(
                    in_ch=out_ch,
                    style_dim=style_dim
                )
            )
            in_ch = out_ch

    def forward(
        self,
        styles,
        input_is_style=False, 
        noise=None,
        randomize_noise=True,
        truncation=1,
        truncation_style=None,
        style_inject_idx=None,
        return_style=None
    ):
        """
            styles : 1개일 수도 있고 2개일 수도 있다. 1개일 경우 [[bs x style_dim]], 2개일 경우 
            [[bs x style_dim], [bs x style_dim]]
            input_is_style : style은 noise와 반대라 생각하면된다. 즉 이게 false이면 noise가 들어왔다는 
            말이기 때문에 mapping network를 거쳐야 한다. 
            truncation : truncation을 안하기위한 비율. 즉, 1에 가까워질수록 truncation안함.
            truncation_style: truncation_mean이라고 이해하는게 더 편하다. 이 값을 실제로 구할 때에는
            여러개의 torch.randn을 평균을 낸다. 위의 truncation과 합쳐져서 이 평균값을 얼마나 사용할지 결정한다.
        """
        if not input_is_style:
            styles = [self.style(s) for s in styles]
        if noise is None:  # noise : 17개의 해상도별 noise
            if randomize_noise: noise = [None] * self.num_layers
            else: noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]
        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_style + truncation * (style - truncation_style)
                )
            styles = style_t
        # 이제 중간 스타일을 어떻게 넣을지 정한다. 스타일이 1개일 경우와 2개일 경우로 나눔. 또한 지금 styles는
        # 2차원 또는 3차원 텐서를 담은 리스트 형태이므로 3차원의 텐서로 변환해준다. 
        if len(styles) < 2:  # 스타일이 1개일 경우에는 그냥 모든 style을 입력에서 받은 스타일로 한다.
            if styles[0].ndim < 3: style = styles[0].unsqueeze(1).repeat(1, self.n_styles, 1)
            else: style = styles[0]
        else:  # 스타일이 2개일 경우에는 1~num_layers 중 무작위로 골라서 스타일을 섞어준다. style regularizer
            if style_inject_idx is None: style_inject_idx = random.randint(1, self.n_styles - 1)
            style1 = styles[0].unsqueeze(1).repeat(1, style_inject_idx, 1)
            style2 = styles[1].unsqueeze(1).repeat(1, self.n_styles - style_inject_idx, 1)
            style = torch.cat([style1, style2], dim=1)  # [bs x n_styles x style_dim]
        
        # forward, 1024기준 noise는 17개의 다른 해상도, style : [bs x 18 x style_dim]
        x = self.cinput(style)  # [bs x n_channels[4] x 4 x 4]
        x = self.conv1(x, style[:, 0], noise=noise[0])
        skip = self.to_rgb1(x, style[:, 1], skip=None)
        i = 1
        for conv_layer1, conv_layer2, noise1, noise2, to_rgb_layer in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            x = conv_layer1(x, style[:, i], noise=noise1)
            x = conv_layer2(x, style[:, i+1], noise=noise2)
            skip = to_rgb_layer(x, style[:, i+2], skip=skip)
            i += 2
        output_img = skip
        if return_style: return output_img, style
        else: return output_img, None        
    def get_style(self, x):
        return self.style(x)
    def mean_style(self, n_style):
        style_in = torch.randn(n_style, self.style_dim).cuda()
        style = self.style(style_in).mean(0, keepdim=True)
        return style       
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        downsample=False,
        blur_kernel=[1,3,3,1],
        bias=True,
        activate=True
    ):
        super().__init__()
        layers = []
        if downsample:
            stride = 2
            padding = 0
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p+1) // 2
            pad1 = p // 2 + 1  # kernel_size=3이고 나머지는 default라면 pad0=0, pad1=1
            layers.append(Blur(blur_kernel, pad=[pad0, pad1]))
        else:
            stride=1
            padding=kernel_size//2
        layers.append(
            EqualConv2d(
                in_ch=in_ch,
                out_ch=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias and not activate
            )
        )
        if activate: layers.append(FusedLeakyReLU(out_ch, bias=bias))
        self.layer = nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)            
class ResBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        blur_kernel=[1,3,3,1]
    ): 
        super().__init__()
        self.conv_layer1 = ConvBlock(in_ch, in_ch, 3)
        self.conv_layer2 = ConvBlock(in_ch, out_ch, 3, downsample=True)
        self.skip_layer = ConvBlock(in_ch, out_ch, 1, downsample=True, activate=False, bias=False)
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        skip = self.skip_layer(x)
        out = (out + skip) / math.sqrt(2)
        return out
class Discriminator(nn.Module):
    def __init__(
        self,
        size,
        channel_multiplier=2,
        blur_kernel=[1,3,3,1]
    ):
        super().__init__()
        self.channels = {
            4: 512,
            8: 512, 
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier
        }
        self.log_size = int(math.log(size, 2))
        convs = []
        in_ch = self.channels[size]
        convs.append(ConvBlock(3, in_ch, 1))
        for mul in range(self.log_size, 2, -1):
            res = 2**(mul-1)
            out_ch = self.channels[res]
            convs.append(ResBlock(in_ch, out_ch, blur_kernel=blur_kernel))
            in_ch = out_ch
        self.convs = nn.Sequential(*convs)
        self.stddev_group = 4
        self.stddev_feat = 1
        self.final_conv = ConvBlock(in_ch + 1, self.channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(self.channels[4] * 4 * 4, self.channels[4], activation="fused_lrelu"),
            EqualLinear(self.channels[4], 1)
        )
    def forward(self, x):
        x = self.convs(x)
        bs, ch, h, w = x.shape
        group = min(bs, self.stddev_group)
        stddev = x.reshape(group, -1, self.stddev_feat, ch // self.stddev_feat, h, w)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2,3,4], keepdim=True).squeeze(2)
        stddev = stddev.repeat(group, 1, h, w)
        x = torch.cat([x, stddev], 1)
        x = self.final_conv(x)
        x = x.reshape(bs, -1)
        return self.final_linear(x)  
 
if __name__ == "__main__":
    bs = 16
    in_ch = 3
    out_ch = 512
    style_dim = 100
    size = 1024
    kernel_size = 3
    n_noise = 18
    sample_x = torch.randn((bs, in_ch, size, size)).cuda()

    #### styleconv test ####
    # sample_styleconv = StyleConv(in_ch, out_ch, kernel_size, style_dim)
    # sample_output = sample_styleconv(sample_x, sample_style)

    #### upsample test #### 
    # sample_upsample = Upsample([1,3,3,1])
    # sample_output = sample_upsample(sample_x)
    
    #### G test ####
    from torch.cuda.amp import autocast
    with autocast():
        sample_style = [torch.randn((bs, style_dim)).cuda()]
        print(sample_style[0].type())
        G = Generator(1024, 100, 8).cuda()
        output_img, output_style = G(sample_style, return_style=True)
        print(output_img.type(), output_style.type())
        print(output_img.shape, output_style.shape)

    #### D test ####
    # D = Discriminator(size)
    # sample_output = D(sample_x)
    # print(sample_output.shape)
    # size = 128
    # D = Discriminator(size)
    # sample_x = torch.randn((bs, in_ch, size, size))
    # sample_output = D(sample_x)
    # print(sample_output.shape)
    
