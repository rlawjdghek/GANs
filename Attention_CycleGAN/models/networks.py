import functools

import torch
import torch.nn as nn
from .base_network import BaseNetwork

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
class Basicblock(nn.Module):
    '''A simple version of residual block.'''
    def __init__(self, in_feat, kernel_size=3, stride=1, padding=1, norm='instance'):
        super(Basicblock, self).__init__()

        norm_layer = get_norm_layer(norm)
        residual = [nn.Conv2d(in_feat, in_feat, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    norm_layer(in_feat),
                    nn.ReLU(True),
                    nn.Conv2d(in_feat, in_feat, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    norm_layer(in_feat)]
        self.residual = nn.Sequential(*residual)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        return self.relu(x + self.residual(x))

class Bottleneck(nn.Module):
    def __init__(self, in_feat, out_feat, depth_bottleneck, stride=1, norm='instance'):
        super(Bottleneck, self).__init__()

        norm_layer = get_norm_layer(norm)
        self.in_equal_out = in_feat == out_feat
        
        self.preact = nn.Sequential(norm_layer(in_feat),
                                    nn.ReLU(inplace=True))

        if self.in_equal_out:
            self.shortcut = nn.MaxPool2d(1, stride=stride)
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False))

        residual = [nn.Conv2d(in_feat, depth_bottleneck, kernel_size=1, stride=1, bias=False),
                    norm_layer(depth_bottleneck),
                    nn.ReLU(True),
                    nn.Conv2d(depth_bottleneck, depth_bottleneck, kernel_size=3, stride=stride, padding=1, bias=False),
                    norm_layer(depth_bottleneck),
                    nn.ReLU(True),
                    nn.Conv2d(depth_bottleneck, out_feat, kernel_size=1, stride=1, bias=False),
                    norm_layer(out_feat)]
        self.residual = nn.Sequential(*residual)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        preact = self.preact(x)
        if self.in_equal_out:
            shortcut = self.shortcut(x)
        else:
            shortcut = self.shortcut(preact)
        return self.relu(shortcut + self.residual(x))

class ResNetGenerator_Att(BaseNetwork):
    '''ResNet-based generator for attention mask prediction.'''
    def __init__(self, in_ch, output_ch=1, ngf=64, norm='instance', residual_mode='basic'):
        super(ResNetGenerator_Att, self).__init__()
        assert residual_mode in ['bottleneck', 'basic']

        norm_layer = get_norm_layer(norm)
        encoder = [nn.Conv2d(in_ch, ngf, kernel_size=7, stride=2, padding=3, bias=False),
                   norm_layer(ngf),
                   nn.ReLU(True),
                   nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=False),
                   norm_layer(ngf*2),
                   nn.ReLU(True)]

        if residual_mode == 'bottleneck':
            encoder += [Bottleneck(ngf*2, ngf*2, ngf*2, norm=norm)]
        else:
            encoder += [Basicblock(ngf*2, norm=norm)]
        self.encoder = nn.Sequential(*encoder)

        self.decoder1 = nn.Sequential(nn.Conv2d(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1, bias=False),
                                      norm_layer(ngf*2),
                                      nn.ReLU(True))

        self.decoder2 = nn.Sequential(nn.Conv2d(ngf*2, ngf, kernel_size=3, stride=1, padding=1, bias=False),
                                      norm_layer(ngf),
                                      nn.ReLU(True),
                                      nn.Conv2d(ngf, output_ch, kernel_size=7, stride=1, padding=3, bias=False),
                                      nn.Sigmoid())
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        encoder = self.encoder(x)
        decoder1 = self.decoder1(self.up2(encoder))
        decoder2 = self.decoder2(self.up2(decoder1))
        return decoder2

class ResNetGenerator_Img(BaseNetwork):
    '''ResNet-based generator for target generation.'''
    def __init__(self, in_ch, out_ch, ngf=64, num_blocks=9, norm='instance', residual_mode='basic'):
        super(ResNetGenerator_Img, self).__init__()
        assert residual_mode in ['bottleneck', 'basic']

        norm_layer = get_norm_layer(norm)
        model = [nn.Conv2d(in_ch, ngf, kernel_size=7, stride=1, padding=3, bias=False),
                 norm_layer(ngf),
                 nn.ReLU(True),
                 nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=False),
                 norm_layer(ngf*2),
                 nn.ReLU(True),
                 nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, bias=False),
                 norm_layer(ngf*4),
                 nn.ReLU(True)]

        for i in range(num_blocks):
            if residual_mode == 'bottleneck':
                model += [Bottleneck(ngf*4, ngf*4, ngf, norm=norm)]
            else:
                model += [Basicblock(ngf*4, norm=norm)]

        model += [nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2,
                                     padding=1, bias=False),
                  norm_layer(ngf*2),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2,
                                     padding=1, bias=False),
                  norm_layer(ngf),
                  nn.ReLU(True),
                  nn.Conv2d(ngf, out_ch, kernel_size=7, stride=1, padding=3, bias=False),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(BaseNetwork):
    '''Discriminator'''
    def __init__(self, in_ch, ndf=64, n_layers=3, norm='instance', transition_rate=0.1):
        super(Discriminator, self).__init__()

        self.transition_rate = transition_rate
        norm_layer = get_norm_layer(norm)
        model = [nn.Conv2d(in_ch, ndf, kernel_size=4, stride=2, padding=1, bias=False), 
                 norm_layer(ndf),
                 nn.LeakyReLU(0.2, True)]

        cur_in, cur_out = ndf, ndf
        for i in range(n_layers):
            cur_in = cur_out
            cur_out =  ndf * min(2**i, 8)
            model += [nn.Conv2d(cur_in, cur_out, kernel_size=4, stride=2, padding=1, bias=False), 
                      norm_layer(cur_out),
                      nn.LeakyReLU(0.2, True)]

        model += [nn.Conv2d(cur_out, 1, kernel_size=4, stride=1, padding=1, bias=False)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return [self.model(x)]