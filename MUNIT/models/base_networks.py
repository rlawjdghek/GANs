import torch.nn as nn
def define_E(args, E_name):
    if E_name == "basic":
        from .networks import Encoder
        # E = Encoder(args.in_ch, args.ngf, args.n_blks, args.n_downsample, args.style_dim)
        E = Encoder(3)
        return E
    else:
        raise NotImplementedError(f"{E_name} is not implemented")
def define_G(args, G_name):
    if G_name == "basic":
        from .networks import Decoder
        # G = Decoder(args.in_ch, args.ngf, args.style_dim, args.n_blks, args.n_upsample)
        G = Decoder(3)
        return G
    else:
        raise NotImplementedError(f"{G_name} is not implemented")
def define_D(args, D_name):
    if D_name == "n_layer":
        from .networks import NLayerDiscriminator
        D = NLayerDiscriminator(args.in_ch, args.ndf, args.D_n_layers, args.D_norm_type)
        return D
    elif D_name == "multi_scale":
        from .networks import MultiScaleDiscriminator
        # D = MultiScaleDiscriminator(args.in_ch, args.ndf, args.D_n_layers, args.D_norm_type, args.n_D)
        D = MultiScaleDiscriminator(args.in_ch)
        return D
    else:
        raise NotImplementedError(f"{D_name} is not implemented")
class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()