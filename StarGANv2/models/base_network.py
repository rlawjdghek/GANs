import torch.nn as nn
from torch.nn import init

# def define_D(args):
#     from .networks_official import Discriminator
#     D = Discriminator(num_domains=3)
#     D.apply(he_init)
#     return D
# def define_G(args):
#     from .networks_official import Generator
#     G = Generator(w_hpf=0)
#     G.apply(he_init)
#     return G
# def define_F(args):
#     from .networks_official import MappingNetwork
#     F = MappingNetwork(num_domains=3)
#     F.apply(he_init)
#     return F
# def define_E(args):
#     from .networks_official import StyleEncoder
#     E = StyleEncoder(num_domains=3)
#     E.apply(he_init)
#     return E
def define_D(args):
    from .networks import Discriminator
    D = Discriminator(args.in_ch, args.img_size, args.n_domains, max_ndf=args.max_ndf)
    D.apply(he_init)
    return D
def define_G(args):
    from .networks import Generator
    G = Generator(args.in_ch, args.in_ch, img_size=args.img_size, style_dim=args.style_dim, max_ngf=args.max_ngf, w_hpf=args.w_hpf)
    G.apply(he_init)
    return G
def define_F(args):
    from .networks import MappingNetwork
    M = MappingNetwork(latent_dim=args.latent_dim, style_dim=args.style_dim, n_domains=args.n_domains)
    M.apply(he_init)
    return M
def define_E(args):
    from .networks import StyleEncoder
    E = StyleEncoder(args.in_ch, img_size=args.img_size, style_dim=args.style_dim, n_domains=args.n_domains, max_nef=args.max_nef)
    E.apply(he_init)
    return E
def count_params(model):
    n_params = 0
    for p in model.parameters():
        n_params += p.numel()
    return n_params
def he_init(module):
    if isinstance(module, nn.Conv2d): 
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
def init_weight(model, init_type="normal", init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, 0.0, init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    model.apply(init_func)
    return model
class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()