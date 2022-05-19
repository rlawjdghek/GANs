import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import functools

class Identity(nn.Module):
    def forward(self, x):
        return x
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError(f"Normalization layer {norm_type} is not implemented!!!"
                                  f"use ['batch', 'instance', 'none']")
    return norm_layer
def get_scheduler(args, optimizer):
    if args.lr_scheduler == "linear":
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + args.start_epoch - args.n_epochs) / float(args.linearlr_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_scheduler == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.steplr_step)
    elif args.lr_scheduler == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif args.lr_scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=0)
    else:
        raise NotImplementedError(f"learning rate scheduler {args.lr_scheduler} is not impleemented!!!"
                                  f"use ['linear', 'step', 'plateau', 'cosine']")
    return scheduler
def init_weights(model, init_type="normal", init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    model.apply(init_func)
    return model
def cal_gradient_penalty(D, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = D(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None
class GANLoss(nn.Module):
    def __init__(self, loss_name, target_real_label=1.0, target_gene_label=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("gene_label", torch.tensor(target_gene_label))
        self.loss_name = loss_name
        if self.loss_name == "lsgan":
            self.loss = nn.MSELoss()
        elif self.loss_name == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.loss_name == "wgangp":
            self.loss = None
        else:
            raise NotImplementedError(f"GAN loss name {self.loss_name} is not implemented!!!!")

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.gene_label
        return target_tensor.expand_as(prediction)
    def forward(self, prediction, target_is_real):
        if self.loss_name == "lsgan" or self.loss_name == "vanilla":
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.loss_name == "wgangp":
            if target_is_real: loss = -prediction.mean()
            else: loss = prediction.mean()
        return loss
class ResNetBlk(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding type {padding_type} is not implemented!!!!"
                                      f"use ['reflect', 'replicate', 'zero']")
        conv_block.append(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias))
        conv_block.append(norm_layer(dim))
        conv_block.append(nn.ReLU(True))
        if use_dropout:
            conv_block.append(nn.Dropout(0.5))

        p = 0
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding type {padding_type} is not implemented!!!!"
                                      f"use ['reflect', 'replicate', 'zero']")
        conv_block.append(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias))
        conv_block.append(norm_layer(dim))
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
class ResNetGenerator(nn.Module):
    def __init__(self, input_ch, output_ch, ngf=64, n_blks=6, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 padding_type="reflect"):
        super().__init__()
        model = []
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model.append(nn.ReflectionPad2d(3))
        model.append(nn.Conv2d(input_ch, ngf, kernel_size=7, padding=0, bias=use_bias))
        model.append(norm_layer(ngf))
        model.append(nn.ReLU(True))

        n_downsample = 2
        for i in range(n_downsample):
            mult = 2 ** i
            model.append(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias))
            model.append(norm_layer(ngf * mult * 2))
            model.append(nn.ReLU(True))
        mult = 2 ** n_downsample
        for i in range(n_blks):
            model.append(ResNetBlk(ngf*mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                     use_bias=use_bias))
        for i in range(n_downsample):
            mult = 2 ** (n_downsample - i)
            model.append(nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                            output_padding=1, bias=use_bias))
            model.append(norm_layer(int(ngf * mult / 2)))
            model.append(nn.ReLU(True))
        model.append(nn.ReflectionPad2d(3))
        model.append(nn.Conv2d(ngf, output_ch, kernel_size=7, padding=0))
        model.append(nn.Tanh())

        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)
class UNetSkipConnectionBlk(nn.Module):
    def __init__(self, inner_ch, output_ch, submodule, input_ch = None, norm_layer=nn.BatchNorm2d,
                 is_innermost=False, is_outermost=False, use_dropout=False):
        '''
        conv 1개와 convT 1개로 이루어진 블록이다. 중간에 submodule이 안쪽의 네트워크이므로 UNet 그림을 그려서 이해하자. innermost부터
        outermost까지 구현된다고 보면 된다.
        :param inner_ch: conv와 convT를 연결하는 # channels. innermost가 아닌 convT에서는 conv의 output의 2배가 되어야한다.
        :param output_ch: output channles
        :param submodule: 중간 레이어. 나중에 UNet을 구성할 때 inner부터 점진적으로 들어간다.
        :param input_ch: input channel. 별로 안중요 한 이유는 어차피 output_ch와 같기 떄문이다.
        :param ngf: number of generator filters
        :param norm_layer: 보통 batchnorm 쓰는듯.
        :param is_innermost: 가장 안쪽인지
        :param is_outermost: 가장 바깥 쪽인지
        '''
        super().__init__()
        self.is_outermost = is_outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if not is_outermost:
            input_ch = output_ch
        down_conv = nn.Conv2d(input_ch, inner_ch, kernel_size=4, stride=2, padding=1, bias=use_bias)
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = norm_layer(inner_ch)
        up_relu = nn.ReLU(True)
        up_norm = norm_layer(output_ch)

        blk = []
        if is_innermost:  # 가장 안쪽 레이어
            up_conv = nn.ConvTranspose2d(inner_ch, output_ch, kernel_size=4, stride=2, padding=1, bias=use_bias)
            blk.append(down_relu)
            blk.append(down_conv)
            blk.append(up_relu)
            blk.append(up_conv)
            blk.append(up_norm)

        elif is_outermost: # 가장 바깥쪽 레이어
            up_conv = nn.ConvTranspose2d(inner_ch * 2, output_ch, kernel_size=4, stride=2, padding=1)
            blk.append(down_conv)
            blk.append(submodule)
            blk.append(up_relu)
            blk.append(up_conv)
            blk.append(nn.Tanh())

        else:  # 중간 레이어들
            up_conv = nn.ConvTranspose2d(inner_ch * 2, output_ch, kernel_size=4, stride=2, padding=1, bias=use_bias)
            blk.append(down_relu)
            blk.append(down_conv)
            blk.append(down_norm)
            blk.append(submodule)
            blk.append(up_relu)
            blk.append(up_conv)
            blk.append(up_norm)
            if use_dropout:
                blk.append(nn.Dropout(0.5))

        self.blk = nn.Sequential(*blk)

    def forward(self, x):
        if self.is_outermost:
            return self.blk(x)
        else:
            return torch.cat([x, self.blk(x)], dim=1)
class UNetGenerator(nn.Module):
    def __init__(self, input_ch, output_ch, ngf=64, n_blks=5, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        blk = UNetSkipConnectionBlk(ngf*8, ngf*8, submodule=None, norm_layer=norm_layer, is_innermost=True)  # innermost
        for _ in range(n_blks - 5):
            blk = UNetSkipConnectionBlk(ngf*8, ngf*8, submodule=blk, norm_layer=norm_layer, use_dropout=use_dropout)
        blk = UNetSkipConnectionBlk(ngf*8, ngf*4, submodule=blk, norm_layer=norm_layer)
        blk = UNetSkipConnectionBlk(ngf*4, ngf*2, submodule=blk, norm_layer=norm_layer)
        blk = UNetSkipConnectionBlk(ngf*2, ngf, submodule=blk, norm_layer=norm_layer)
        self.model = UNetSkipConnectionBlk(ngf, output_ch, input_ch=input_ch, submodule=blk, norm_layer=norm_layer,
                                           is_outermost=True)
    def forward(self, x):
        return self.model(x)
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_ch, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        model.append(nn.Conv2d(input_ch, ndf, kernel_size=4, stride=2, padding=1))
        model.append(nn.LeakyReLU(0.2, True))
        nf_mult = 1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8)
            model.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias))
            model.append(norm_layer(ndf*nf_mult))
            model.append(nn.LeakyReLU(0.2, True))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias))
        model.append(norm_layer(ndf*nf_mult))
        model.append(nn.LeakyReLU(0.2, True))

        model.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)
class PixelDiscriminator(nn.Module):
    def __init__(self, input_ch, ndf=64, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = []
        model.append(nn.Conv2d(input_ch, ndf, kernel_size=1, stride=1, padding=0))
        model.append(nn.LeakyReLU(0.2, True))
        model.append(nn.Conv2d(ndf, ndf*2, kernel_size=1, stride=1, padding=0, bias=use_bias))
        model.append(norm_layer(ndf*2))
        model.append(nn.LeakyReLU(0.2, True))
        model.append(nn.Conv2d(ndf*2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias))

        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)
def define_G(G_name,
             input_ch,
             output_ch,
             ngf,
             norm_type="batch",
             init_type="normal",
             init_gain=0.02,
             use_dropout=False):
    norm_layer = get_norm_layer(norm_type)
    if G_name == "resnet_6blks":
        G = ResNetGenerator(input_ch=input_ch,
                            output_ch=output_ch,
                            ngf=ngf,
                            n_blks=6,
                            norm_layer=norm_layer,
                            use_dropout=use_dropout)
    elif G_name == "resnet_9blks":
        G = ResNetGenerator(input_ch=input_ch,
                            output_ch=output_ch,
                            ngf=ngf,
                            n_blks=9,
                            norm_layer=norm_layer,
                            use_dropout=use_dropout)
    elif G_name == "unet_128":  # unet은 blk한개당 downsample 1번
        G = UNetGenerator(input_ch=input_ch,
                          output_ch=output_ch,
                          ngf=ngf,
                          n_blks=7,
                          norm_layer=norm_layer,
                          use_dropout=use_dropout)
    elif G_name == "unet_256":
        G = UNetGenerator(input_ch=input_ch,
                          output_ch=output_ch,
                          ngf=ngf,
                          n_blks=8,
                          norm_layer=norm_layer,
                          use_dropout=use_dropout)
    elif G_name == "unet_512":
        G = UNetGenerator(input_ch=input_ch,
                          output_ch=output_ch,
                          ngf=ngf,
                          n_blks=9,
                          norm_layer=norm_layer,
                          use_dropout=use_dropout)
    else:
        raise NotImplementedError(f"Generator {G_name} is not implemented!!!!"
                                  f"use ['resnet_6blks','resnet_9blks','unet_128','unet_256','unet_512']")
    return init_weights(G, init_type=init_type, init_gain=init_gain)
def define_D(D_name,
             input_ch,
             ndf,
             n_layers,
             norm_type="batch",
             init_type="normal",
             init_gain=0.02):
    norm_layer = get_norm_layer(norm_type)
    if D_name == "basic":
        D = NLayerDiscriminator(input_ch=input_ch,
                                ndf=ndf,
                                n_layers=3,
                                norm_layer=norm_layer)
    elif D_name == "n_layers":
        D = NLayerDiscriminator(input_ch=input_ch,
                                ndf=ndf,
                                n_layers=n_layers,
                                norm_layer=norm_layer)
    elif D_name == "pixel":
        D = NLayerDiscriminator(input_ch=input_ch,
                                ndf=ndf,
                                norm_layer=norm_layer)
    else:
        raise NotImplementedError(f"Discriminator {D_name} is not implemented!!!!"
                                  f"use ['basic', 'n_layers', 'pixel']")
    return init_weights(D, init_type=init_type, init_gain=init_gain)

if __name__ == "__main__":
    sample_input = torch.randn((4, 3, 512, 512)).cuda()
    # G_resnet6 = define_G("resnet_6blks", 3, 3, 64).cuda()
    # print(f"G renset6: {G_resnet6(sample_input).shape}")
    # G_resnet9 = define_G("resnet_9blks", 3, 3, 64).cuda()
    # print(f"G renset9: {G_resnet9(sample_input).shape}")
    # G_unet128 = define_G("unet_128", 3, 3, 64).cuda()
    # print(f"G unet128: {G_unet128(sample_input).shape}")
    # G_unet256 = define_G("unet_256", 3, 3, 64).cuda()
    # print(f"G unet256: {G_unet256(sample_input).shape}")
    # G_unet512 = define_G("unet_512", 3, 3, 64).cuda()
    # print(f"G unet512: {G_unet512(sample_input).shape}")

    D_basic = define_D("basic", 3, 64, 5).cuda()
    print(f"D basic: {D_basic(sample_input).shape}")
    D_n_layers = define_D("n_layers", 3, 64, 5).cuda()
    print(f"D n layers: {D_n_layers(sample_input).shape}")
    D_pixel = define_D("pixel", 3, 64, 5).cuda()
    print(f"D pixel: {D_pixel(sample_input).shape}")

