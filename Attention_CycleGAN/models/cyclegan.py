import itertools
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from .base_model import *
from .base_network import define_G, define_D

class AttentionCycleGAN(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.G_AB = define_G(args, args.G_AB_name).cuda(args.local_rank)
        self.G_BA = define_G(args, args.G_BA_name).cuda(args.local_rank)
        self.G_attn_A = define_G(args, args.G_attn_A_name).cuda(args.local_rank)
        self.G_attn_B = define_G(args, args.G_attn_B_name).cuda(args.local_rank)
        if not args.is_test:
            self.D_A = define_D(args, args.D_AB_name).cuda(args.local_rank)
            self.D_B = define_D(args, args.D_BA_name).cuda(args.local_rank)

        if args.use_DDP:
            self.G_AB = DistributedDataParallel(self.G_AB, device_ids=[args.local_rank])
            self.G_BA = DistributedDataParallel(self.G_BA, device_ids=[args.local_rank])
            self.G_attn_A = DistributedDataParallel(self.G_attn_A, device_ids=[args.local_rank])
            self.G_attn_B = DistributedDataParallel(self.G_attn_B, device_ids=[args.local_rank])
            if not args.is_test:
                self.D_A = DistributedDataParallel(self.D_A, device_ids=[args.local_rank])
                self.D_B = DistributedDataParallel(self.D_B, device_ids=[args.local_rank])
        if args.use_mask_for_D:  # D에 mask된 이미지를 넣을 땐, attn으로 보간된 이미지 (gene_A)와 함꼐 mask 이미지 (attn_A)까지 넣는다
            self.gene_A_pool = ImageMaskPool(args.pool_size)
            self.gene_B_pool = ImageMaskPool(args.pool_size)
        else:
            self.gene_A_pool = ImagePool(args.pool_size)
            self.gene_B_pool = ImagePool(args.pool_size)
        
        if not args.is_test:
            self.criterion_GAN = GANLoss(args.gan_loss_name, target_real_label=args.target_real_label, target_gene_label=args.target_gene_label).cuda(args.local_rank)
            self.criterion_cycle = nn.L1Loss()
            self.criterion_ID = nn.L1Loss()
            if not args.no_vgg:
                self.criterion_VGG = VGGLoss(args.local_rank)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=args.G_lr, betas=args.G_betas)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=args.D_lr, betas=args.D_betas)

            self.scheduler_G = get_scheduler(args, optimizer=self.optimizer_G)
            self.scheduler_D = get_scheduler(args, optimizer=self.optimizer_D)   
    def set_input(self, real_A, real_B):
        self.real_A = real_A
        self.real_B = real_B
    def forward_G(self):
        self.attn_A = self.G_attn_A(self.real_A)
        self.raw_gene_B = self.G_AB(self.real_A)
        if self.args.is_test:
            self.attn_A *= (self.attn_A > self.args.attn_thres)  # test에는 attn mask에서 픽셀값이 thres보다 작으면 그냥 0으로 친다.
        self.gene_B = self.attn_A * self.raw_gene_B + (1 - self.attn_A) * self.real_A

        self.attn_B = self.G_attn_B(self.real_B)
        self.raw_gene_A = self.G_BA(self.real_B)
        if self.args.is_test:
            self.attn_B *= (self.attn_B > self.args.attn_thres)
        self.gene_A = self.attn_B * self.raw_gene_A + (1 - self.attn_B) * self.real_B
        
        self.recon_attn_B = self.G_attn_B(self.gene_B)
        self.raw_recon_A = self.G_BA(self.gene_B)
        self.recon_A = self.recon_attn_B * self.raw_recon_A + (1 - self.recon_attn_B) * self.gene_B

        self.recon_attn_A = self.G_attn_A(self.gene_A)
        self.raw_recon_B = self.G_AB(self.gene_A)
        self.recon_B = self.recon_attn_A * self.raw_recon_B + (1 - self.recon_attn_A) * self.gene_A
        self.attn_A_viz = (self.attn_A - 0.5) * 2
        self.attn_B_viz = (self.attn_B - 0.5) * 2
    def get_loss_G(self):
        gene_A = self.gene_A  # 여기서 따로 빼놓는 이유는 D에서 query에 넣을때 보간된 이미지를 넣어야한다.
        gene_B = self.gene_B
        if self.args.use_mask_for_D: 
            gene_A *= self.attn_B  # mask된 이미지를 넣을땐 foreground만 살린다. 
            gene_B *= self.attn_A  # mask된 이미지를 넣을땐 foreground만 살린다. 
        loss_G_A = self.criterion_GAN(self.D_B(gene_B), True)
        loss_G_B = self.criterion_GAN(self.D_A(gene_A), True)

        loss_cycle_A = self.criterion_cycle(self.recon_A, self.real_A) * self.args.lambda_A
        loss_cycle_B = self.criterion_cycle(self.recon_B, self.real_B) * self.args.lambda_B

        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B
        return loss_G           
    def get_loss_D(self):
        if self.args.use_mask_for_D:
            gene_A, attn_B = self.gene_A_pool.query(self.gene_A, self.attn_B)
            gene_A *= attn_B
            gene_B, attn_A = self.gene_B_pool.query(self.gene_B, self.attn_A)
            gene_B *= attn_A
        else:
            gene_A = self.gene_A_pool.query(self.gene_A)
            gene_B = self.gene_B_pool.query(self.gene_B)
        loss_D_real = self.criterion_GAN(self.D_A(self.real_A), True)
        loss_D_gene = self.criterion_GAN(self.D_A(self.gene_A.detach()), False)
        loss_D_A = loss_D_real + loss_D_gene

        loss_D_real = self.criterion_GAN(self.D_B(self.real_B), True)
        loss_D_gene = self.criterion_GAN(self.D_B(self.gene_B.detach()), False)
        loss_D_B = loss_D_real + loss_D_gene
        
        loss_D = loss_D_A + loss_D_B
        return loss_D
    def train(self, cur_epoch):
        self.set_requires_grad([self.G_AB, self.G_BA], requires_grad=True)
        self.set_requires_grad([self.D_A, self.D_B], requires_grad=False)
        if cur_epoch < self.args.stop_attn_learning_epoch:  # 일정 에폭 이상이면 attn model은 학습 안함.
            self.set_requires_grad([self.G_attn_A, self.G_attn_B], requires_grad=True)
        else:
            self.set_requires_grad([self.G_attn_A, self.G_attn_B], requires_grad=False)
        
        self.forward_G()
        self.loss_G = self.get_loss_G()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        self.set_requires_grad([self.G_AB, self.G_BA, self.G_attn_A, self.G_attn_B], requires_grad=False)
        self.set_requires_grad([self.D_A, self.D_B], requires_grad=True)
        self.loss_D = self.get_loss_D()
        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()
    def save(self, to_path):
        save_state_dict = {}
        if self.args.use_DDP:
            save_state_dict["G_AB"] = self.G_AB.module.state_dict()
            save_state_dict["G_BA"] = self.G_BA.module.state_dict()
            save_state_dict["D_A"] = self.D_A.module.state_dict()
            save_state_dict["D_B"] = self.D_B.module.state_dict()
        else:
            save_state_dict["G_AB"] = self.G_AB.state_dict()
            save_state_dict["G_BA"] = self.G_BA.state_dict()
            save_state_dict["D_A"] = self.D_A.state_dict()
            save_state_dict["D_B"] = self.D_B.state_dict()
        save_state_dict["optimizer_G"] = self.optimizer_G.state_dict()
        save_state_dict["optimizer_D"] = self.optimizer_D.state_dict()
        if self.args.local_rank == 0:
            torch.save(save_state_dict, to_path)
    def load(self, load_path):
        load_state_dict = torch.load(load_path)
        self.G_AB.load_state_dict(load_state_dict["G_AB"])
        print(f"===Model Load=== model is succesfully loaded from {load_path}")
    def inference(self, real_A):
        with torch.no_grad():
            gene_B, _ = self.G_AB(real_A)
        return gene_B