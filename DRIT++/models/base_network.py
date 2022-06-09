import torch.nn as nn

def define_E(args, E_type):
    if E_type == "content":
        from .networks import EncoderContent
        E = EncoderContent(args.in_ch_A, args.in_ch_B, nef=args.nef_content)
        return E
    elif E_type == "style":
        from .networks import EncoderStyle
        E = EncoderStyle(args.in_ch_A, args.in_ch_B, style_dim=args.style_dim, nef=args.nef_style)
        return E
def define_G(args, G_name):
    if G_name == "res_4blks":
        from .networks import Generator
        G = Generator(args.out_ch_A, args.out_ch_B, ngf=args.ngf, style_dim=args.style_dim)
        return G
def define_D(args, D_name):
    if D_name == "multi_scale":
        from .networks import MultiScaleDiscriminator
        D = MultiScaleDiscriminator(args.in_ch, args.ndf, args.n_D, args.n_layer, norm="None")
        return D
    elif D_name == "content":
        from .networks import DiscriminatorContent
        D = DiscriminatorContent(args.ndf_content)
        return D

# def define_E(args, E_type):
#     if E_type == "content":
#         from .networks_official import E_content
#         E = E_content(3,3)
#         return E
#     elif E_type == "style":
#         from .networks_official import E_attr
#         E = E_attr(3,3)
#         return E
# def define_G(args, G_name):
#     if G_name == "res_4blks":
#         from .networks_official import Generator
#         G = Generator(3,3, 8)
#         return G
# def define_D(args, D_name):
#     if D_name == "multi_scale":
#         from .networks_official import MultiScaleDis
#         D = MultiScaleDis(3)
#         return D
#     elif D_name == "content":
#         from .networks_official import Dis_content
#         D = Dis_content()
#         return D


class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()