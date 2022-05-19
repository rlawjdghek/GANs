import torch.nn as nn

def define_module_G(args):
    if args.G_name == "official":
        from .vid2vidG import Vid2VidModelG
        G = Vid2VidModelG(args)
    return G
class BaseModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args