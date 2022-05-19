from abc import ABC
from itertools import chain

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler

from utils.utils import adjust_learning_rate
from .base_model import BaseModel
from .networks import Generator, Discriminator, RhoClipper

class UGATITModel(BaseModel):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        local_rank = args.local_rank
        self.G_AB = Generator(args.in_ch, args.out_ch, n_down=args.G_n_down, ngf=args.ngf, n_blocks=args.G_n_blocks, img_size=args.img_size, light=args.light).cuda(args.local_rank)
        self.G_BA = Generator(args.in_ch, args.out_ch, n_down=args.G_n_down, ngf=args.ngf, n_blocks=args.G_n_blocks, img_size=args.img_size, light=args.light).cuda(args.local_rank)
        self.D_local_A = Discriminator(args.in_ch, ndf=args.ndf, n_layers=args.D_local_n_layers).cuda(local_rank)
        self.D_local_B = Discriminator(args.in_ch, ndf=args.ndf, n_layers=args.D_local_n_layers).cuda(local_rank)
        self.D_global_A = Discriminator(args.in_ch, ndf=args.ndf, n_layers=args.D_global_n_layers).cuda(local_rank)
        self.D_global_B = Discriminator(args.in_ch, ndf=args.ndf, n_layers=args.D_global_n_layers).cuda(local_rank)

        self.G_AB = DistributedDataParallel(self.G_AB, device_ids=[local_rank])
        self.G_BA = DistributedDataParallel(self.G_BA, device_ids=[local_rank])
        self.D_local_A = DistributedDataParallel(self.D_local_A, device_ids=[local_rank])
        self.D_local_B = DistributedDataParallel(self.D_local_B, device_ids=[local_rank])
        self.D_global_A = DistributedDataParallel(self.D_global_A, device_ids=[local_rank])
        self.D_global_B = DistributedDataParallel(self.D_global_B, device_ids=[local_rank])
        self.criterion_L1 = nn.L1Loss()
        self.criterion_MSE = nn.MSELoss()
        self.criterion_BCE = nn.BCEWithLogitsLoss()

        self.G_optim = optim.Adam(chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=args.G_lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
        self.D_optim = optim.Adam(chain(self.D_local_A.parameters(), self.D_local_B.parameters(), self.D_global_A.parameters(), self.D_global_B.parameters()), lr=args.D_lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)

        self.rho_clipper = RhoClipper(0, 1)
        self.scaler = GradScaler()
    def set_input(self, img_A, img_B):
        self.real_A = img_A
        self.real_B = img_B
    def train(self, cur_iter):
        adjust_learning_rate(self.args, self.G_optim, cur_iter)
        adjust_learning_rate(self.args, self.D_optim, cur_iter)
        #### update D ####
        self.set_requires_grad([self.D_global_A, self.D_global_B, self.D_local_A, self.D_local_B], requires_grad=True)
        self.set_requires_grad([self.G_AB, self.G_BA], requires_grad=False)
        #### A에 대해서 먼저. ####
        #### 고려해야할 것: 1. global vs local 2. cam vs non-cam 3. real vs gene => 총 8개
        gene_A, _, _ = self.G_BA(self.real_B)
        real_global_logit, real_global_cam_logit, _ = self.D_global_A(self.real_A)
        real_local_logit, real_local_cam_logit, _ = self.D_local_A(self.real_A)
        gene_global_logit, gene_global_cam_logit, _ = self.D_global_A(gene_A)
        gene_local_logit, gene_local_cam_logit, _ = self.D_local_A(gene_A)

        #### label 먼저 다 만들어 두기. 
        global_one_labels = torch.ones_like(real_global_logit).cuda(self.args.local_rank)
        global_zero_labels = torch.zeros_like(global_one_labels).cuda(self.args.local_rank)
        local_one_labels = torch.ones_like(real_local_logit).cuda(self.args.local_rank)
        local_zero_labels = torch.zeros_like(local_one_labels).cuda(self.args.local_rank)
        global_one_cam_labels = torch.ones_like(real_global_cam_logit).cuda(self.args.local_rank)
        global_zero_cam_labels = torch.zeros_like(global_one_cam_labels).cuda(self.args.local_rank)
        local_one_cam_labels = torch.ones_like(real_local_cam_logit).cuda(self.args.local_rank)
        local_zero_cam_labels = torch.ones_like(local_one_cam_labels).cuda(self.args.local_rank)

        D_adv_loss_global = self.criterion_MSE(real_global_logit, global_one_labels) + self.criterion_MSE(gene_global_logit, global_zero_labels)
        D_adv_loss_global_cam = self.criterion_MSE(real_global_cam_logit, global_one_cam_labels) + self.criterion_MSE(gene_global_cam_logit, global_zero_cam_labels)
        D_adv_loss_local = self.criterion_MSE(real_local_logit, local_one_labels) + self.criterion_MSE(gene_local_logit, local_zero_labels)
        D_adv_loss_local_cam = self.criterion_MSE(real_local_cam_logit, local_one_cam_labels) + self.criterion_MSE(gene_local_cam_logit, local_zero_cam_labels)
        D_loss_A = self.args.lambda_adv * (D_adv_loss_global + D_adv_loss_global_cam + D_adv_loss_local + D_adv_loss_local_cam)

        #### B에 대해서도 해야됨.
        gene_B, _, _ = self.G_AB(self.real_A)
        real_global_logit, real_global_cam_logit, _ = self.D_global_B(self.real_B)
        real_local_logit, real_local_cam_logit, _ = self.D_local_B(self.real_B)
        gene_global_logit, gene_global_cam_logit, _ = self.D_global_B(gene_B)
        gene_local_logit, gene_local_cam_logit, _ = self.D_local_B(gene_B)

        D_adv_loss_global = self.criterion_MSE(real_global_logit, global_one_labels) + self.criterion_MSE(gene_global_logit, global_zero_labels)
        D_adv_loss_global_cam = self.criterion_MSE(real_global_cam_logit, global_one_cam_labels) + self.criterion_MSE(gene_global_cam_logit, global_zero_cam_labels)
        D_adv_loss_local = self.criterion_MSE(real_local_logit, local_one_labels) + self.criterion_MSE(gene_local_logit, local_zero_labels)
        D_adv_loss_local_cam = self.criterion_MSE(real_local_cam_logit, local_one_cam_labels) + self.criterion_MSE(gene_local_cam_logit, local_zero_cam_labels)
        D_loss_B = self.args.lambda_adv * (D_adv_loss_global + D_adv_loss_global_cam + D_adv_loss_local + D_adv_loss_local_cam)
        self.D_loss = D_loss_A + D_loss_B
        self.D_optim.zero_grad()
        self.D_loss.backward()
        self.D_optim.step()

        #### update G ####
        self.set_requires_grad([self.G_AB, self.G_BA], requires_grad=True)
        self.set_requires_grad([self.D_global_A, self.D_global_B, self.D_local_A, self.D_local_B], requires_grad=False)
        #### A먼저 ####
        gene_A, gene_A_cam_logit, _ = self.G_BA(self.real_B)
        gene_B, gene_B_cam_logit, _ = self.G_AB(self.real_A)
        recon_A, _, _ = self.G_BA(gene_B)
        recon_B, _, _ = self.G_AB(gene_A)
        id_A, id_A_cam_logit, _ = self.G_BA(self.real_A)

        G_cam_one_labels = torch.ones_like(gene_A_cam_logit, requires_grad=False).cuda(self.args.local_rank)
        G_cam_zero_labels = torch.zeros_like(gene_A_cam_logit, requires_grad=False).cuda(self.args.local_rank)
        gene_global_logit, gene_global_cam_logit, _ = self.D_global_A(gene_A)
        gene_local_logit, gene_local_cam_logit, _ = self.D_local_A(gene_A)
        
        G_adv_loss_global = self.criterion_MSE(gene_global_logit, global_one_labels)
        G_adv_loss_global_cam = self.criterion_MSE(gene_global_cam_logit, global_one_cam_labels)
        G_adv_loss_local = self.criterion_MSE(gene_local_logit, local_one_labels)
        G_adv_loss_local_cam = self.criterion_MSE(gene_local_cam_logit, local_one_cam_labels)

        G_recon_loss = self.criterion_L1(recon_A, self.real_A)
        G_id_loss = self.criterion_L1(id_A, self.real_A)
        G_cam_loss = self.criterion_BCE(gene_A_cam_logit, G_cam_one_labels) + self.criterion_BCE(id_A_cam_logit, G_cam_zero_labels)
        G_loss_A = self.args.lambda_adv * (G_adv_loss_global + G_adv_loss_global_cam + G_adv_loss_local + G_adv_loss_local_cam) + self.args.lambda_cycle * G_recon_loss + self.args.lambda_id * G_id_loss + self.args.lambda_cam * G_cam_loss

        #### B ####
        id_B, id_B_cam_logit, _ = self.G_AB(self.real_B)

        gene_global_logit, gene_global_cam_logit, _ = self.D_global_B(gene_B)
        gene_local_logit, gene_local_cam_logit, _ = self.D_local_B(gene_B)

        G_adv_loss_global = self.criterion_MSE(gene_global_logit, global_one_labels)
        G_adv_loss_global_cam = self.criterion_MSE(gene_global_cam_logit, global_one_cam_labels)
        G_adv_loss_local = self.criterion_MSE(gene_local_logit, local_one_labels)
        G_adv_loss_local_cam = self.criterion_MSE(gene_local_cam_logit, local_one_cam_labels)

        G_recon_loss = self.criterion_L1(recon_B, self.real_B)
        G_id_loss = self.criterion_L1(id_B, self.real_B)
        G_cam_loss = self.criterion_BCE(gene_B_cam_logit, G_cam_one_labels) + self.criterion_BCE(id_B_cam_logit, G_cam_zero_labels)       
        G_loss_B = self.args.lambda_adv * (G_adv_loss_global + G_adv_loss_global_cam + G_adv_loss_local + G_adv_loss_local_cam) + self.args.lambda_cycle * G_recon_loss + self.args.lambda_id * G_id_loss + self.args.lambda_cam * G_cam_loss
        self.G_loss = G_loss_A + G_loss_B
        self.G_optim.zero_grad()
        self.G_loss.backward()
        self.G_optim.step()
        self.G_AB.apply(self.rho_clipper)
        self.G_BA.apply(self.rho_clipper)
    def inference(self, img_A, img_B):
        with torch.no_grad():
            gene_A, _, G_heatmap_A = self.G_BA(img_B)
            gene_B, _, G_heatmap_B = self.G_AB(img_A)
            _, _, D_global_heatmap_A = self.D_global_A(img_A)
            _, _, D_global_heatmap_B = self.D_global_B(img_B)
            _, _, D_local_heatmap_A = self.D_local_A(img_A)
            _, _, D_local_heatmap_B = self.D_local_B(img_B)
        return gene_A, G_heatmap_A, gene_B, G_heatmap_B, D_global_heatmap_A, D_global_heatmap_B, D_local_heatmap_A, D_local_heatmap_B
    def to_train(self):
        self.G_AB.train()
        self.G_BA.train()
        self.D_global_A.train()
        self.D_global_B.train()
        self.D_local_A.train()
        self.D_local_B.train()
    def to_eval(self):
        self.G_AB.eval()
        self.G_BA.eval()
        self.D_global_A.eval()
        self.D_global_B.eval()
        self.D_local_A.eval()
        self.D_local_B.eval()
    def save(self, cur_iter, to_path):
        state_dict = {}
        state_dict["cur_iter"] = cur_iter
        state_dict["G_AB"] = self.G_AB.module.state_dict()
        state_dict["G_BA"] = self.G_BA.module.state_dict()
        state_dict["D_global_A"] = self.D_global_A.module.state_dict()
        state_dict["D_global_B"] = self.D_global_B.module.state_dict()
        state_dict["D_local_A"] = self.D_local_A.module.state_dict()
        state_dict["D_local_B"] = self.D_local_B.module.state_dict()
        state_dict["optimizer_G"] = self.G_optim.state_dict()
        state_dict["optimizer_D"] = self.D_optim.state_dict()
        torch.save(state_dict, to_path)
        
    def load(self):
        pass