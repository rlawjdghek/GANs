import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from .base_network import define_E, define_G, define_D
from .base_model import BaseModel, get_scheduler, BCELoss, BCEHalfLoss, L2RegLoss, gaussian_weights_init

class DRIT_pp(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.D_A = define_D(args, args.D_name).cuda(args.local_rank)
        self.D_A2 = define_D(args, args.D_name).cuda(args.local_rank)
        self.D_B = define_D(args, args.D_name).cuda(args.local_rank)
        self.D_B2 = define_D(args, args.D_name).cuda(args.local_rank)
        self.D_c = define_D(args, D_name='content').cuda(args.local_rank)
        self.E_c = define_E(args, E_type="content").cuda(args.local_rank)
        self.E_s = define_E(args, E_type="style").cuda(args.local_rank)
        self.G = define_G(args, args.G_name).cuda(args.local_rank)

        self.D_A.apply(gaussian_weights_init)
        self.D_A2.apply(gaussian_weights_init)
        self.D_B.apply(gaussian_weights_init)
        self.D_B2.apply(gaussian_weights_init)
        self.D_c.apply(gaussian_weights_init)
        self.E_c.apply(gaussian_weights_init)
        self.E_s.apply(gaussian_weights_init)
        self.G.apply(gaussian_weights_init)

        if args.use_DDP:
            self.D_A = DDP(self.D_A, device_ids=[args.local_rank])
            self.D_A2 = DDP(self.D_A2, device_ids=[args.local_rank])
            self.D_B = DDP(self.D_B, device_ids=[args.local_rank])
            self.D_B2 = DDP(self.D_B2, device_ids=[args.local_rank])
            self.D_c = DDP(self.D_c, device_ids=[args.local_rank])
            self.E_c = DDP(self.E_c, device_ids=[args.local_rank])
            self.E_s = DDP(self.E_s, device_ids=[args.local_rank])
            self.G = DDP(self.G, device_ids=[args.local_rank])

        self.optimizer_D_A = optim.Adam(self.D_A.parameters(), lr=args.D_lr, betas=args.betas, weight_decay=1e-4)
        self.optimizer_D_A2 = optim.Adam(self.D_A2.parameters(), lr=args.D_lr, betas=args.betas, weight_decay=1e-4)
        self.optimizer_D_B = optim.Adam(self.D_B.parameters(), lr=args.D_lr, betas=args.betas, weight_decay=1e-4)
        self.optimizer_D_B2 = optim.Adam(self.D_B2.parameters(), lr=args.D_lr, betas=args.betas, weight_decay=1e-4)
        self.optimizer_D_c = optim.Adam(self.D_c.parameters(), lr=args.D_lr, betas=args.betas, weight_decay=1e-4)
        self.optimizer_E_c = optim.Adam(self.E_c.parameters(), lr=args.E_lr, betas=args.betas, weight_decay=1e-4)
        self.optimizer_E_s = optim.Adam(self.E_s.parameters(), lr=args.E_lr, betas=args.betas, weight_decay=1e-4)
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=args.G_lr, betas=args.betas, weight_decay=1e-4)
        
        self.criterion_L1 = nn.L1Loss()
        self.criterion_BCE = BCELoss()
        self.criterion_BCEHalf = BCEHalfLoss()
        self.criterion_L2_reg = L2RegLoss()

        self.scheduler_D_A = get_scheduler(args, self.optimizer_D_A)
        self.scheduler_D_A2 = get_scheduler(args, self.optimizer_D_A2)
        self.scheduler_D_B = get_scheduler(args, self.optimizer_D_B)
        self.scheduler_D_B2 = get_scheduler(args, self.optimizer_D_B2)
        self.scheduler_D_c = get_scheduler(args, self.optimizer_D_c)
        self.scheduler_E_c = get_scheduler(args, self.optimizer_E_c)
        self.scheduler_E_s = get_scheduler(args, self.optimizer_E_s)
        self.scheduler_G = get_scheduler(args, self.optimizer_G)
    def set_input(self, real_A, real_B, z1, z2):
        self.real_A = real_A
        self.real_B = real_B
        self.z1 = z1
        self.z2 = z2
    def train(self, cur_iter):
        if cur_iter % self.args.D_c_iter != 0 and cur_iter < self.args.epoch_iter - 2:
            self.update_D_c()
        else:
            self.update_DEG()
    def update_DEG(self):
        BS = self.real_A.shape[0]
        half_BS = BS // 2
        self.real_A1 = self.real_A[:half_BS]  # 실제파트
        self.real_A2 = self.real_A[half_BS:]  # random style 부분
        self.real_B1 = self.real_B[:half_BS]
        self.real_B2 = self.real_B[half_BS:]

        self.c_A, self.c_B = self.E_c(self.real_A1, self.real_B1)
        self.s_A, self.s_B = self.E_s(self.real_A1, self.real_B1)
        
        input_c_A = torch.cat([self.c_B, self.c_A, self.c_B, self.c_B], 0)
        input_s_A = torch.cat([self.s_A, self.s_A, self.z1, self.z2], 0)
        input_c_B = torch.cat([self.c_A, self.c_B, self.c_A, self.c_A], 0)
        input_s_B = torch.cat([self.s_B, self.s_B, self.z1, self.z2], 0)

        if self.args.use_DDP:
            output_A = self.G.module.forward_A(input_c_A, input_s_A)
            output_B = self.G.module.forward_B(input_c_B, input_s_B)
        else:
            output_A = self.G.forward_A(input_c_A, input_s_A)
            output_B = self.G.forward_B(input_c_B, input_s_B)
        self.gene_A, self.recon_A, self.gene_random_A1, self.gene_random_A2 = torch.split(output_A, half_BS, 0)
        self.gene_B, self.recon_B, self.gene_random_B1, self.gene_random_B2 = torch.split(output_B, half_BS, 0)
        
        self.gene_c_B, self.gene_c_A = self.E_c(self.gene_A, self.gene_B)
        self.gene_s_A, self.gene_s_B = self.E_s(self.gene_A, self.gene_B)
        
        if self.args.use_DDP:
            self.cycle_A = self.G.module.forward_A(self.gene_c_A, self.gene_s_A)
            self.cycle_B = self.G.module.forward_B(self.gene_c_B, self.gene_s_B)
        else:
            self.cycle_A = self.G.forward_A(self.gene_c_A, self.gene_s_A)
            self.cycle_B = self.G.forward_B(self.gene_c_B, self.gene_s_B)

        self.gene_random_A1_s, self.gene_random_B1_s = self.E_s(self.gene_random_A1, self.gene_random_B1)

        #### D update ####
        # D_A update
        pred_real = self.D_A(self.real_A1)
        pred_gene = self.D_A(self.gene_A.detach())
        self.loss_D_A1 = self.criterion_BCE(pred_real, True) + self.criterion_BCE(pred_gene, False)
        self.optimizer_D_A.zero_grad()
        self.loss_D_A1.backward()
        self.optimizer_D_A.step()
        
        # D_A2 update
        pred_real = self.D_A2(self.real_A2)
        pred_gene = self.D_A2(self.gene_random_A1.detach())
        loss_D_A2_1 = self.criterion_BCE(pred_real, True) + self.criterion_BCE(pred_gene, False)

        pred_real = self.D_A2(self.real_A2)
        pred_gene = self.D_A2(self.gene_random_A2.detach())
        loss_D_A2_2 = self.criterion_BCE(pred_real, True) + self.criterion_BCE(pred_gene, False)
        self.loss_D_A2 = loss_D_A2_1 + loss_D_A2_2
        self.optimizer_D_A2.zero_grad()
        self.loss_D_A2.backward()
        self.optimizer_D_A2.step()

        # D_B update
        pred_real = self.D_B(self.real_B1)
        pred_gene = self.D_B(self.gene_B.detach())
        self.loss_D_B1 = self.criterion_BCE(pred_real, True) + self.criterion_BCE(pred_gene, False)
        self.optimizer_D_B.zero_grad()
        self.loss_D_B1.backward()
        self.optimizer_D_B.step()

        # D_B2 update
        pred_real = self.D_B2(self.real_B2)
        pred_gene = self.D_B2(self.gene_random_B1.detach())
        loss_D_B2_1 = self.criterion_BCE(pred_real, True) + self.criterion_BCE(pred_gene, False)

        pred_real = self.D_B2(self.real_B2)
        pred_gene = self.D_B2(self.gene_random_B2.detach())
        loss_D_B2_2 = self.criterion_BCE(pred_real, True) + self.criterion_BCE(pred_gene, False)
        self.loss_D_B2 = loss_D_B2_1 + loss_D_B2_2
        self.optimizer_D_B2.zero_grad()
        self.loss_D_B2.backward()
        self.optimizer_D_B2.step()

        # D_c update
        pred_c_A = self.D_c(self.c_A.detach())
        pred_c_B = self.D_c(self.c_B.detach())
        self.loss_D_c = self.criterion_BCE(pred_c_B, True) + self.criterion_BCE(pred_c_A, False)
        self.optimizer_D_c.zero_grad()
        self.loss_D_c.backward()
        nn.utils.clip_grad_norm_(self.D_c.parameters(), 5)
        self.optimizer_D_c.step()
        
        #### E, G update ####
        # content adv
        pred_A_c = self.D_c(self.c_A)
        pred_B_c = self.D_c(self.c_B)
        loss_G_GAN_c = self.criterion_BCEHalf(pred_A_c) + self.criterion_BCEHalf(pred_B_c)
        
        # adv 
        pred_gene_A = self.D_A(self.gene_A)
        pred_gene_B = self.D_B(self.gene_B)
        loss_G_GAN = self.criterion_BCE(pred_gene_A, True) + self.criterion_BCE(pred_gene_B, True)

        # style KL 
        loss_G_kl_s_A = self.criterion_L2_reg(self.s_A) * self.args.lambda_l2reg
        loss_G_kl_s_B = self.criterion_L2_reg(self.s_B) * self.args.lambda_l2reg
        loss_G_kl_s = loss_G_kl_s_A + loss_G_kl_s_B

        # content KL 
        loss_G_kl_c_A = self.criterion_L2_reg(self.c_A) * self.args.lambda_l2reg
        loss_G_kl_c_B = self.criterion_L2_reg(self.c_B) * self.args.lambda_l2reg
        loss_G_kl_c = loss_G_kl_c_A + loss_G_kl_c_B

        loss_G_recon_A = self.criterion_L1(self.real_A1, self.recon_A) * self.args.lambda_recon
        loss_G_recon_B = self.criterion_L1(self.real_B1, self.recon_B) * self.args.lambda_recon
        loss_G_recon = loss_G_recon_A + loss_G_recon_B

        loss_G_cycle_A = self.criterion_L1(self.real_A1, self.cycle_A) * self.args.lambda_cycle
        loss_G_cycle_B = self.criterion_L1(self.real_B1, self.cycle_B) * self.args.lambda_cycle
        loss_G_cycle = loss_G_cycle_A + loss_G_cycle_B


        self.loss_G = loss_G_GAN_c + loss_G_GAN + loss_G_kl_s + loss_G_kl_c + loss_G_recon + loss_G_cycle

        self.optimizer_E_c.zero_grad()
        self.optimizer_E_s.zero_grad()
        self.optimizer_G.zero_grad()
        self.loss_G.backward(retain_graph=True)
        self.optimizer_E_c.step()
        self.optimizer_E_s.step()
        self.optimizer_G.step()

        BS = self.real_A.shape[0]
        half_BS = BS // 2
        self.real_A1 = self.real_A[:half_BS]  # 실제파트
        self.real_A2 = self.real_A[half_BS:]  # random style 부분
        self.real_B1 = self.real_B[:half_BS]
        self.real_B2 = self.real_B[half_BS:]

        self.c_A, self.c_B = self.E_c(self.real_A1, self.real_B1)
        self.s_A, self.s_B = self.E_s(self.real_A1, self.real_B1)
        
        input_c_A = torch.cat([self.c_B, self.c_A, self.c_B, self.c_B], 0)
        input_s_A = torch.cat([self.s_A, self.s_A, self.z1, self.z2], 0)
        input_c_B = torch.cat([self.c_A, self.c_B, self.c_A, self.c_A], 0)
        input_s_B = torch.cat([self.s_B, self.s_B, self.z1, self.z2], 0)

        if self.args.use_DDP:
            output_A = self.G.module.forward_A(input_c_A, input_s_A)
            output_B = self.G.module.forward_B(input_c_B, input_s_B)
        else:
            output_A = self.G.forward_A(input_c_A, input_s_A)
            output_B = self.G.forward_B(input_c_B, input_s_B)
        self.gene_A, self.recon_A, self.gene_random_A1, self.gene_random_A2 = torch.split(output_A, half_BS, 0)
        self.gene_B, self.recon_B, self.gene_random_B1, self.gene_random_B2 = torch.split(output_B, half_BS, 0)
        
        self.gene_c_B, self.gene_c_A = self.E_c(self.gene_A, self.gene_B)
        self.gene_s_A, self.gene_s_B = self.E_s(self.gene_A, self.gene_B)
        
        if self.args.use_DDP:
            self.cycle_A = self.G.module.forward_A(self.gene_c_A, self.gene_s_A)
            self.cycle_B = self.G.module.forward_B(self.gene_c_B, self.gene_s_B)
        else:
            self.cycle_A = self.G.forward_A(self.gene_c_A, self.gene_s_A)
            self.cycle_B = self.G.forward_B(self.gene_c_B, self.gene_s_B)

        self.gene_random_A1_s, self.gene_random_B1_s = self.E_s(self.gene_random_A1, self.gene_random_B1)

        # random variable
        pred_gene_random_A1 = self.D_A2(self.gene_random_A1)
        pred_gene_random_A2 = self.D_A2(self.gene_random_A2)
        pred_gene_random_B1 = self.D_B2(self.gene_random_B1)
        pred_gene_random_B2 = self.D_B2(self.gene_random_B2)
        loss_G_GAN_random = self.criterion_BCE(pred_gene_random_A1, True) + \
                      self.criterion_BCE(pred_gene_random_A2, True) + \
                      self.criterion_BCE(pred_gene_random_B1, True) + \
                      self.criterion_BCE(pred_gene_random_B2, True)
        ms_A = torch.mean(torch.abs(self.gene_random_A2 - self.gene_random_A1)) / torch.mean(torch.abs(self.z2 - self.z1))
        ms_B = torch.mean(torch.abs(self.gene_random_B2 - self.gene_random_B1)) / torch.mean(torch.abs(self.z2 - self.z1))

        eps = 1e-5
        loss_ms_A = 1 / (ms_A+eps)
        loss_ms_B = 1 / (ms_B+eps)
        loss_ms = loss_ms_A + loss_ms_B

        loss_z_L1_A = torch.mean(torch.abs(self.gene_random_A1_s - self.z1)) * self.args.lambda_z_L1
        loss_z_L1_B = torch.mean(torch.abs(self.gene_random_B1_s - self.z1)) * self.args.lambda_z_L1
        loss_z_L1 = loss_z_L1_A + loss_z_L1_B

        self.loss_G_random = loss_G_GAN_random + loss_ms + loss_z_L1
        
        self.optimizer_E_c.zero_grad()
        self.optimizer_G.zero_grad()
        self.loss_G_random.backward()
        self.optimizer_E_c.step()
        self.optimizer_G.step()

    def update_D_c(self):  # content를 구분하는 discriminator D_c 훈련
        BS = self.real_A.shape[0]
        half_BS = BS // 2
        self.real_A1 = self.real_A[:half_BS]
        self.real_B1 = self.real_B[:half_BS]
        self.c_A, self.c_B = self.E_c(self.real_A1, self.real_B1)
        pred_c_A = self.D_c(self.c_A.detach())
        pred_c_B = self.D_c(self.c_B.detach())
        self.loss_D_c = self.criterion_BCE(pred_c_B, True) + self.criterion_BCE(pred_c_A, False)
        self.optimizer_D_c.zero_grad()
        self.loss_D_c.backward()
        nn.utils.clip_grad_norm_(self.D_c.parameters(), 5)
        self.optimizer_D_c.step()
        
        

        
        
