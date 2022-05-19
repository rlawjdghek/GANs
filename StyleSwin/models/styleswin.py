from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd
from torch.nn.parallel import DistributedDataParallel

from .base_model import BaseModel
from .networks import Generator, Discriminator
from utils.distributed import reduce_loss_dict
class StyleSwinModel(BaseModel):
    def __init__(self, args, logger):
        BaseModel.__init__(self, args, logger)
        self.G = Generator(args.style_dim, args.n_mapping_networks)
        self.G_ema = Generator(args.style_dim, args.n_mapping_networks)
        self.D = Discriminator()
        self.G.cuda(args.local_rank)
        self.D.cuda(args.local_rank)
        self.G_ema.cuda(args.local_rank)
        self.G_ema.eval()
        self.accumulate(self.G_ema, self.G, 0)
        if args.use_DDP:
            self.G = DistributedDataParallel(self.G, device_ids=[args.local_rank])
            self.D = DistributedDataParallel(self.D, device_ids=[args.local_rank])
        
        self.G_reg_ratio = args.G_reg_every / (args.G_reg_every + 1)
        self.D_reg_ratio = args.D_reg_every / (args.D_reg_every + 1)
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=args.G_lr * self.G_reg_ratio, betas=(args.beta1 ** self.G_reg_ratio, args.beta2 ** self.G_reg_ratio))        
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=args.D_lr * self.D_reg_ratio, betas=(args.beta1 ** self.D_reg_ratio, args.beta2 ** self.D_reg_ratio))
        self.D_loss_val = 0
        self.G_loss_val = 0
        self.D_r1_loss = torch.tensor(0.0).cuda(args.local_rank)
        self.loss_dict = {}
        
    def set_input(self, img, img2, z, z2):
        self.real_img = img
        self.z = z
        self.real_img2 = img2
        self.z2 = z2
    def train(self, cur_iter):
        # train D
        self.set_requires_grad(self.G, requires_grad=False)
        self.set_requires_grad(self.D, requires_grad=True)

        gene_img = self.G(self.z)
        gene_pred = self.D(gene_img)
        real_pred = self.D(self.real_img)
        D_loss_1 = F.softplus(-real_pred)
        D_loss_2 = F.softplus(gene_pred)
        D_loss = (D_loss_1.mean() + D_loss_2.mean())
        self.optimizer_D.zero_grad()
        D_loss.backward()
        nn.utils.clip_grad_norm_(self.D.parameters(), 5.0)
        self.optimizer_D.step()

        if (cur_iter-1) % self.args.D_reg_every == 0:
            self.real_img.requires_grad = True
            real_pred = self.D(self.real_img)
            grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=self.real_img, create_graph=True)
            r1_loss = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
            self.D_r1_loss = self.args.r1 / 2 * r1_loss * self.args.D_reg_every
            self.optimizer_D.zero_grad()
            self.D_r1_loss.backward()
            self.optimizer_D.step()
        self.loss_dict["D_loss"] = D_loss
        self.loss_dict["D_r1_loss"] = self.D_r1_loss
        
        # train G
        self.set_requires_grad(self.G, requires_grad=True)
        self.set_requires_grad(self.D, requires_grad=False)
        gene_img = self.G(self.z2)
        gene_pred = self.D(gene_img)
        G_loss = F.softplus(-gene_pred).mean()
        self.optimizer_G.zero_grad()
        G_loss.backward()
        self.optimizer_G.step()
        self.loss_dict["G_loss"] = G_loss

        # ema
        if self.args.use_DDP: self.accumulate(self.G_ema, self.G.module, self.args.G_ema_decay)
        else: self.accumulate(self.G_ema, self.G, self.args.G_ema_decay)

        loss_reduced = reduce_loss_dict(self.loss_dict)
        self.D_loss_val = loss_reduced["D_loss"].mean().item()
        self.G_loss_val = loss_reduced["G_loss"].mean().item()
        self.D_r1_loss_val = loss_reduced["D_r1_loss"].mean().item()
    def inference(self):
        self.G_ema.eval()
        with torch.no_grad():
            z = torch.randn((1, self.args.style_dim)).cuda(self.args.local_rank)
            gene_img = self.G_ema(z)
        return gene_img
    def save(self, to_path):
        if self.args.use_DDP:
            save_state_dict = {
                "G" : self.G.module.state_dict(),
                "D" : self.D.module.state_dict(),
                "G_ema" : self.G_ema.state_dict(),
                "optimizer_G" : self.optimizer_G.state_dict(),
                "optimizer_D" : self.optimizer_D.state_dict() 
            }
        else:
            save_state_dict = {
                "G" : self.G.state_dict(),
                "D" : self.D.state_dict(),
                "G_ema" : self.G_ema.state_dict(),
                "optimizer_G" : self.optimizer_G.state_dict(),
                "optimizer_D" : self.optimizer_D.state_dict() 
            }
        torch.save(save_state_dict, to_path)
    def load(self):
        pass
    def _weight_init(self, m):
        cls_name = m.__class__.__name__
        if cls_name.find("Conv2d") != -1:
            if self.args.layer_init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif self.args.layer_init_type == "orth":
                nn.init.orthogonal_(m.weight.data)
            elif self.args.layer_init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight.data, 1.0)
            else: raise NotImplementedError(f"initialize {self.args.layer_init_type} is not implemented!!!!")
        elif cls_name.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    
        
