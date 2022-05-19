import random

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim

from .base_model import BaseModel
from .networks import Generator, Discriminator
from utils.utils import AverageMeter
from non_leaking import AdaptiveAugment, augment
from op import conv2d_gradfix
from distributed import *

class StyleGAN2Model(BaseModel):
    def __init__(self, args, logger):
        BaseModel.__init__(self, args, logger)
        self.G = Generator(
            size=self.args.size, 
            style_dim=self.args.style_dim, 
            n_mlp=self.args.n_mlp,
            channel_multiplier=self.args.channel_multiplier
            ).cuda(args.local_rank)
        self.D = Discriminator(
            size=self.args.size,
            channel_multiplier=self.args.channel_multiplier
        ).cuda(args.local_rank)
        self.G_ema = Generator(
            size=self.args.size,
            style_dim=self.args.style_dim,
            n_mlp=self.args.n_mlp,
            channel_multiplier=self.args.channel_multiplier
        ).cuda(args.local_rank)
        self.G_ema.eval()
        self.accumulate(self.G_ema, self.G, 0)  # 여기서는 복사의 의미??
        self.G = nn.parallel.DistributedDataParallel(
            self.G, 
            device_ids=[args.local_rank], 
            output_device=args.local_rank,
            broadcast_buffers=False
        )
        self.D = nn.parallel.DistributedDataParallel(
            self.D,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False
        )
        
        reg_ratio_G = args.reg_every_G / (args.reg_every_G + 1)
        reg_ratio_D = args.reg_every_D / (args.reg_every_D + 1)
        self.optimizer_G = optim.Adam(
            self.G.parameters(),
            lr=args.lr * reg_ratio_G,
            betas=(0 ** reg_ratio_G, 0.99 ** reg_ratio_G)
        )
        self.optimizer_D = optim.Adam(
            self.D.parameters(),
            lr=args.lr * reg_ratio_D,
            betas=(0**reg_ratio_D, 0.99 ** reg_ratio_D)
        )
        self.start_iter = 1
        if args.cp_path is not None:
            self.logger.write(f"load models from {args.cp_path}")
            self.load_model()
        
        #### metrics ####
        self.r1_loss = torch.tensor(0.0).cuda(self.args.local_rank)
        self.path_loss = torch.tensor(0.0).cuda(self.args.local_rank)
        self.path_lengths = torch.tensor(0.0).cuda(self.args.local_rank)
        self.mean_path_length = 0
        self.mean_path_length_avg = 0
        self.loss_dict = {}
        self.accum = 0.5 ** (32 / (10 * 1000))
        self.ada_aug_p = self.args.augment_p if self.args.augment_p > 0 else 0.0
        self.r_t_stat = 0
        if self.args.augment and self.args.augment_p == 0:
            self.ada_augment = AdaptiveAugment(self.args.ada_target)
        self.sample_z = torch.randn(self.args.n_samples, self.args.style_dim).cuda(args.local_rank)
    def __name__(self):
        return "StyleGAN2Model"     
    def accumulate(self, model1, model2, decay=0.999):
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())
        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
    def set_input(self, real_img):
        self.real_img = real_img
    def forward_G(self, return_style=False):  #### TODO : 삭제
        self.gene_img, self.style = self.G(self.noise, return_style=return_style)
    def train(self, iter):
        #### train D ####
        self.set_requires_grad(self.G, requires_grad=False)
        self.set_requires_grad(self.D, requires_grad=True)
        
        self.noise = self.mixing_noise(
        bs=self.args.batch_size, 
        style_dim=self.args.style_dim,
        prob=self.args.mixing            
        )
        self.forward_G()   
        if self.args.augment:
            self.real_img_aug, _ = augment(self.real_img, self.ada_aug_p)
            self.gene_img, _ = augment(self.gene_img, self.ada_aug_p)
        else: self.real_img_aug = self.real_img
        self.real_pred = self.D(self.real_img_aug)
        self.gene_pred = self.D(self.gene_img)
        self.loss_D = self.logistic_loss_D()
        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()

        self.loss_dict["loss_D"] = self.loss_D
        self.loss_dict["real_loss_D"] = self.real_loss_D.mean()
        self.loss_dict["gene_loss_D"] = self.gene_loss_D.mean()


        if self.args.augment and self.args.augment_p == 0:
            self.ada_aug_p = self.ada_augment.tune(self.real_pred)
            self.r_t_stat = self.ada_augment.r_t_stat
        d_regularize = iter % self.args.reg_every_D == 0
        if d_regularize:
            self.real_img.requires_grad = True
            
            if self.args.augment: self.real_img_aug, _ = augment(self.real_img, self.ada_aug_p)
            else: self.real_img_aug = self.real_img
            self.real_pred = self.D(self.real_img_aug)
            self.r1_loss = self.r1_loss_D()
            self.optimizer_D.zero_grad()
            (self.args.r1 / 2 * self.r1_loss * self.args.reg_every_D + 0 * self.real_pred[0]).backward()
            self.optimizer_D.step()
        self.loss_dict["r1_loss"] = self.r1_loss

        #### train G ####
        self.set_requires_grad(self.G, True)
        self.set_requires_grad(self.D, False)
        
        #### adv loss ####
        self.noise = self.mixing_noise(
        bs=self.args.batch_size, 
        style_dim=self.args.style_dim,
        prob=self.args.mixing            
        )
        
        self.forward_G()
        self.gene_pred = self.D(self.gene_img)
        self.loss_G = self.nonsaturating_loss_G()
        self.loss_dict["loss_G"] = self.loss_G
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        #### G path regularizer ####
        g_regularize = iter % self.args.reg_every_G == 0
        if g_regularize:
            path_batch_size = max(1, self.args.batch_size // self.args.path_batch_shrink)
            self.noise = self.mixing_noise(
                bs=path_batch_size,
                style_dim=self.args.style_dim,
                prob=self.args.mixing
            )
            
            self.forward_G(return_style=True)
            self.path_loss, self.mean_path_length, self.path_lengths = self.path_regularize_G()
            self.weighted_path_loss = self.args.w_path_regularize * self.args.reg_every_G * self.path_loss
            self.optimizer_G.zero_grad()
            if self.args.path_batch_shrink: self.weighted_path_loss += 0 * self.gene_img[0,0,0,0]
            self.weighted_path_loss.backward()
            self.optimizer_G.step()
            self.mean_path_length_avg = reduce_sum(self.mean_path_length).item() / self.args.world_size
        self.loss_dict["path"] = self.path_loss
        self.loss_dict["path_length"] = self.path_lengths.mean()
        self.accumulate(self.G_ema, self.G.module, self.accum)
        self.loss_reduced = reduce_loss_dict(self.loss_dict)
        self.loss_val_D = self.loss_reduced["loss_D"].mean().item()
        self.loss_val_G = self.loss_reduced["loss_G"].mean().item()
        self.r1_loss_val = self.loss_reduced["r1_loss"].mean().item()
        self.path_loss_val = self.loss_reduced["path"].mean().item()
        self.real_loss_val_D = self.loss_reduced["real_loss_D"].mean().item()
        self.gene_loss_val_D = self.loss_reduced["gene_loss_D"].mean().item()
        self.path_length_val = self.loss_reduced["path_length"].mean().item()
    def inference(self):
        with torch.no_grad():
            self.G_ema.eval()
            sample, _ = self.G_ema([self.sample_z])
            self.G_ema.train()
        return sample
    def make_noise(self, bs, style_dim, n_noise):
        if n_noise == 1: return torch.randn(bs, style_dim).cuda(self.args.local_rank)
        else: return torch.randn(n_noise, bs, style_dim).cuda(self.args.local_rank).unbind(0)
    def mixing_noise(self, bs, style_dim, prob):
        if prob > 0 and random.random() < prob: return self.make_noise(bs, style_dim, 2)
        else: return [self.make_noise(bs, style_dim, 1)]    
    def logistic_loss_D(self):
        self.real_loss_D = F.softplus(-self.real_pred)
        self.gene_loss_D = F.softplus(self.gene_pred)
        return self.real_loss_D.mean() + self.gene_loss_D.mean()
    def r1_loss_D(self):
        with conv2d_gradfix.no_weight_gradients():
            grad_real, = autograd.grad(outputs=self.real_pred.sum(), inputs=self.real_img, create_graph=True)
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty
    def nonsaturating_loss_G(self):
        return F.softplus(-self.gene_pred).mean()
    def path_regularize_G(self, decay=0.01):
        noise = torch.randn_like(self.gene_img) / math.sqrt(self.gene_img.shape[2] * self.gene_img.shape[3])
        grad, = autograd.grad(
            outputs=(self.gene_img * noise).sum(), inputs=self.style, create_graph=True
        )
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
        path_mean = self.mean_path_length + decay * (path_lengths.mean() - self.mean_path_length)
        path_penalty = (path_lengths - path_mean).pow(2).mean()
        return path_penalty, path_mean.detach(), path_lengths
    def load(self):
        cp = torch.load(self.args.cp_path)
        self.G.module.load_state_dict(cp["G"], map_location={"cuda:0":f"cuda:{self.args.local_rank}"})
        self.D.module.load_state_dict(cp["D"], map_location={"cuda:0":f"cuda:{self.args.local_rank}"})
        self.G_ema.module.load_state_dict(cp["G_ema"], map_location={"cuda:0":f"cuda:{self.args.local_rank}"})
        self.optimizer_G.load_state_dict(cp["optimizer_G"])
        self.optimizer_D.load_state_dict(cp["optimizer_D"])
        self.start_iter = cp["iter"]
    def save(self, iter, to_path):
        assert self.args.local_rank == 0
        cp = {}
        cp["iter"] = iter
        cp["G"] = self.G.module.state_dict()
        cp["D"] = self.D.module.state_dict()
        cp["G_ema"] = self.G_ema.state_dict()
        cp["optimizer_G"] = self.optimizer_G.state_dict()
        cp["optimizer_D"] = self.optimizer_D.state_dict()
        torch.save(cp, to_path)


        
        


        
    
