from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel
from models.functional import *
from models.networks import Generator, Discriminator
# from models.no_args_gen import Generator

class TransGANModel(BaseModel):
    def __init__(self, args, logger):
        BaseModel.__init__(self, args, logger)
        #### model ####
        self.G = Generator(
            latent_dim=args.latent_dim, 
            bottom_width=args.bottom_width, 
            embedding_dim=args.G_embedding_dim, 
            depths=args.G_depths, 
            out_ch=args.G_out_ch, 
            n_heads=args.G_n_heads, 
            norm_type=args.G_norm_type, 
            window_size=args.G_window_size, 
            mlp_ratio=args.G_mlp_ratio,
            act_type=args.G_act_type
        )
        self.D = Discriminator(
            in_ch=args.D_in_ch, 
            diff_aug=args.diff_aug, 
            n_heads=args.D_n_heads, 
            n_cls=args.n_cls,
            img_size=args.img_size,
            patch_size=args.D_patch_size, 
            embedding_dim=args.D_embedding_dim,
            window_size=args.D_window_size, 
            depth=args.D_depths,
            mlp_ratio=args.D_mlp_ratio,
            act_type=args.D_act_type,
            norm_type=args.D_norm_type
        )
        self.weight_init()
        self.G.cuda(args.local_rank)
        self.D.cuda(args.local_rank)

        self.G = nn.parallel.DistributedDataParallel(self.G, device_ids=[args.local_rank])
        self.D = nn.parallel.DistributedDataParallel(self.D, device_ids=[args.local_rank])
        #### optimizer ####
        if args.G_optim == "adam": 
            # self.G_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), args.G_lr, (args.G_beta1, args.G_beta2))
            self.G_optim = torch.optim.Adam(self.G.parameters(), args.G_lr, (args.G_beta1, args.G_beta2))
        elif args.G_optim == "adamW": 
            self.G_optim = torch.optim.AdamW(self.G.parameters(), args.G_lr, weight_decay=args.G_weight_decay)
        if args.D_optim == "adam": 
            # self.D_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), args.D_lr, (args.D_beta1, args.D_beta2))
            self.D_optim = torch.optim.Adam(self.D.parameters(), args.D_lr, (args.D_beta1, args.D_beta2))
        elif args.D_optim == "adamW": 
            self.D_optim = torch.optim.AdamW(self.D.parameters(), args.D_lr, weight_decay=args.D_weight_decay)

        G_avg = deepcopy(self.G).cpu()
        self.G_avg_param = copy_params(G_avg)
        del G_avg
        self.fixed_z = torch.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))
    def set_input(self, img, z):
        self.real_img = img
        self.z = z
    def train(self, cur_iter):
        self.real_pred = self.D(self.real_img)
        self.gene_img = self.G(self.z).detach()
        self.gene_pred = self.D(self.gene_img)
        if self.args.loss_type == "hinge": 
            self.D_loss = torch.mean(F.relu(1 + self.real_pred)) + torch.mean(F.relu(1 - self.gene_pred))
        elif self.args.loss_type == "standard":
            real_label = torch.ones((self.real_pred.shape[0]), dtype=torch.float32).cuda(self.args.local_rank)
            gene_label = torch.zeros((self.real_pred.shape[0]), dtype=torch.float32).cuda(self.args.local_rank)
            self.real_pred = F.sigmoid(self.real_pred.squeeze(1))
            self.gene_pred = F.sigmoid(self.gene_pred.squeeze(1))
            self.D_real_loss = F.binary_cross_entropy(self.real_pred, real_label)
            self.D_gene_loss = F.binary_cross_entropy(self.gene_pred, gene_label)
            self.D_loss = self.D_real_loss + self.D_gene_loss
        elif self.args.loss_type == "lsgan":
            real_label = torch.ones_like(self.real_pred, dtype=torch.float32).cuda(self.args.local_rank)
            gene_label = torch.zeros_like(self.real_pred, dtype=torch.float32).cuda(self.args.local_rank)
            self.D_real_loss = F.mse_loss(self.real_pred, real_label)
            self.D_gene_loss = F.mse_loss(self.gene_pred, gene_label)
            self.D_loss = self.D_real_loss + self.D_gene_loss
        elif self.args.loss_type == "wgangp": 
            gp = compute_gradient_penalty(self.D, self.real_img, self.gene_img.detach(), self.args.wgangp_phi)
            self.D_loss = -torch.mean(self.real_pred) + torch.mean(self.gene_pred) + (gp * 10) / (self.args.wgangp_phi ** 2)
        elif self.args.loss_type == "wgangp-mode":
            gp = compute_gradient_penalty(self.D, self.real_img, self.gene_img.detach(), self.args.wgangp_phi)
            self.D_loss = -torch.mean(self.real_pred) + torch.mean(self.gene_pred) + (gp * 10) / (self.args.wgangp_phi ** 2)
        elif self.args.loss_type == "wgangp-eps": 
            gp = compute_gradient_penalty(self.D, self.real_img, self.gene_img.detach(), self.args.wgangp_phi)
            self.D_loss = -torch.mean(self.real_pred) + torch.mean(self.gene_pred) + (gp * 10) / (self.args.wgangp_phi ** 2)
            self.D_loss += (torch.mean(self.real_pred) ** 2) * 1e-3
        else: raise NotImplementedError(f"D loss type {self.args.loss_type} is not implemented!!!!")        
        self.D_loss = self.D_loss / float(self.args.accumulated_times)
        self.D_loss.backward()


        if cur_iter % self.args.accumulated_times == 0:
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), 5.0)
            self.D_optim.step() 
            self.D_optim.zero_grad()
        #### training G ####
        if (cur_iter-1) % (self.args.D_iter_per_G * self.args.accumulated_times) == 0:  # wgan을 사용하므로 G는 D 4번에 1번씩 된다.
            for accumulated_idx in range(self.args.G_accumulated_times):
                gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (self.args.batch_size*2, self.args.latent_dim)))
                self.gene_img = self.G(gen_z)
                self.gene_pred = self.D(self.gene_img)
                if self.args.loss_type == "standard":
                    real_label = torch.full((self.args.batch_size*2,), 1., dtype=torch.float, device=self.real_img.get_device())
                    gene_pred = nn.Sigmoid()(self.gene_pred.view(-1))
                    g_loss = nn.BCELoss()(gene_pred.view(-1), real_label)
                if self.args.loss_type == "lsgan":
                    if isinstance(self.gene_pred, list):
                        g_loss = 0
                        for fake_validity_item in self.gene_pred:
                            real_label = torch.full((fake_validity_item.shape[0],fake_validity_item.shape[1]), 1., dtype=torch.float, device=self.real_img.get_device())
                            g_loss += nn.MSELoss()(fake_validity_item, real_label)
                    else:
                        real_label = torch.full((self.gene_pred.shape[0],self.gene_pred.shape[1]), 1., dtype=torch.float, device=self.real_img.get_device())
                        g_loss = nn.MSELoss()(self.gene_pred, real_label)
                elif self.args.loss_type == 'wgangp-mode':
                    fake_image1, fake_image2 = self.gen_img[:self.args.batch_size], self.gen_img[self.args.batch_size:]
                    z_random1, z_random2 = gen_z[:self.args.batch_size], gen_z[self.args.batch_size:]
                    lz = torch.mean(torch.abs(fake_image2 - fake_image1)) / torch.mean(
                    torch.abs(z_random2 - z_random1))
                    eps = 1 * 1e-5
                    loss_lz = 1 / (lz + eps)
                    g_loss = -torch.mean(self.gene_pred) + loss_lz
                else:
                    self.G_loss = -torch.mean(self.gene_pred)
                self.G_loss = self.G_loss / float(self.args.G_accumulated_times)
                self.G_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), 5.0)
            self.G_optim.step()   
            self.G_optim.zero_grad()

            #### moving average weight
            ema_n_img = self.args.ema_k_img * 1000
            cur_n_img = self.args.batch_size * self.args.world_size * cur_iter
            if self.args.ema_warmup != 0:
                ema_n_img = min(ema_n_img, cur_n_img * self.args.ema_warmup)
                ema_beta = 0.5 ** (float(self.args.batch_size * self.args.world_size) / max(ema_n_img, 1e-8))
            else: ema_beta = self.args.ema
            for p, avg_p in zip(self.G.parameters(), self.G_avg_param):
                cpu_p = deepcopy(p)
                avg_p.mul_(ema_beta).add_(1.0-ema_beta, cpu_p.cpu().data)  # ema_beta=0.9999라면 average에 0.9999의 가중치
                del cpu_p
    def weight_init(self):
        self.G.apply(self._weight_init)
        self.D.apply(self._weight_init)
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
    def inference(self):
        self.G.eval()
        with torch.no_grad():
            z = torch.randn((self.args.eval_batch_size, self.args.latent_dim))
            sample_img = self.G(z)
        self.G.train()
        return sample_img
    def save(self, to_path): 
        save_dict = {
            "G": self.G.module.state_dict(),
            "D": self.D.module.state_dict(),
            "G_optim": self.G_optim.state_dict(),
            "D_optim": self.D_optim.state_dict()
        }
        if self.args.local_rank==0:
            torch.save(save_dict, to_path)
    def load(self): pass