import torch.nn as nn
def define_D(args, D_name):
    if D_name == "basic":
        from .networks import Discriminator
        D = Discriminator(args.in_ch)
    return D
def define_G(args, G_name):
    if G_name == "basic_attn":
        from .networks import ResNetGenerator_Att
        G = ResNetGenerator_Att(args.in_ch)
    elif G_name == "res_9blks":
        from .networks import ResNetGenerator_Img
        G = ResNetGenerator_Img(args.in_ch, args.out_ch, num_blocks=9)
    elif G_name == "res_6blks":
        from .networks import ResNetGenerator_Img
        G = ResNetGenerator_Img(args.in_ch, args.out_ch, num_blocks=6)
    return G   

class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
