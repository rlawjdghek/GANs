import itertools

from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
import torch

from .base_model import BaseModel, GANLoss, weights_init_normal
from .base_networks import define_E, define_G, define_D
class MUNIT(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.E_A = define_E(args, args.E_A_name).cuda(args.local_rank)
        self.E_B = define_E(args, args.E_B_name).cuda(args.local_rank)
        self.G_A = define_G(args, args.G_A_name).cuda(args.local_rank)
        self.G_B = define_G(args, args.G_B_name).cuda(args.local_rank)
        self.D_A = define_D(args, args.D_A_name).cuda(args.local_rank)
        self.D_B = define_D(args, args.D_B_name).cuda(args.local_rank)

        self.E_A.apply(weights_init_normal)
        self.E_B.apply(weights_init_normal)
        self.G_A.apply(weights_init_normal)
        self.G_B.apply(weights_init_normal)
        self.D_A.apply(weights_init_normal)
        self.D_B.apply(weights_init_normal)
    
        if args.use_DDP:
            self.E_A = DistributedDataParallel(self.E_A, device_ids=[args.local_rank], find_unused_parameters=True)
            self.E_B = DistributedDataParallel(self.E_B, device_ids=[args.local_rank], find_unused_parameters=True)
            self.G_A = DistributedDataParallel(self.G_A, device_ids=[args.local_rank], find_unused_parameters=True)
            self.G_B = DistributedDataParallel(self.G_B, device_ids=[args.local_rank], find_unused_parameters=True)
            self.D_A = DistributedDataParallel(self.D_A, device_ids=[args.local_rank], find_unused_parameters=True)
            self.D_B = DistributedDataParallel(self.D_B, device_ids=[args.local_rank], find_unused_parameters=True)
        self.criterion_GAN = GANLoss()
        self.criterion_recon = nn.L1Loss()
        
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.E_A.parameters(), self.E_B.parameters(), self.G_A.parameters(), self.G_B.parameters()), lr=args.G_lr, betas=args.G_betas)
        self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr=args.D_lr, betas=args.D_betas)
        self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr=args.D_lr, betas=args.D_betas)
    def set_input(self, real_A, real_B, z_A, z_B):
        self.real_A = real_A
        self.real_B = real_B
        self.z_A = z_A
        self.z_B = z_B
    def get_loss_G(self):
        c_A, s_A = self.E_A(self.real_A)
        c_B, s_B = self.E_B(self.real_B)
        recon_A = self.G_A(c_A, s_A)
        recon_B = self.G_B(c_B, s_B)
        gene_A = self.G_A(c_B, self.z_A)
        gene_B = self.G_B(c_A, self.z_B)
        self.gene_A = gene_A  # Discriminator 용
        self.gene_B = gene_B  # Discriminator 용
        gene_c_B, gene_s_A = self.E_A(gene_A)
        gene_c_A, gene_s_B = self.E_B(gene_B)
        
        if self.args.lambda_cycle > 0:
            cycle_A = self.G_A(gene_c_B, s_A)
            cycle_B = self.G_B(gene_c_A, s_B)
        loss_G_GAN_A = self.criterion_GAN(self.D_A(gene_A), target_is_real=True) * self.args.lambda_GAN
        loss_G_GAN_B = self.criterion_GAN(self.D_B(gene_B), target_is_real=True) * self.args.lambda_GAN
        # loss_G_GAN_A = self.compute_loss(self.D_A(gene_A), 1) * self.args.lambda_GAN
        # loss_G_GAN_B = self.compute_loss(self.D_B(gene_B), 1) * self.args.lambda_GAN
        loss_G_recon_A = self.criterion_recon(recon_A, self.real_A) * self.args.lambda_recon
        loss_G_recon_B = self.criterion_recon(recon_B, self.real_B) * self.args.lambda_recon
        loss_G_recon_s_A = self.criterion_recon(gene_s_A, self.z_A) * self.args.lambda_style
        loss_G_recon_s_B = self.criterion_recon(gene_s_B, self.z_B) * self.args.lambda_style
        loss_G_recon_c_A = self.criterion_recon(gene_c_A, c_A.detach()) * self.args.lambda_content
        loss_G_recon_c_B = self.criterion_recon(gene_c_B, c_B.detach()) * self.args.lambda_content
        if self.args.lambda_cycle > 0:
            loss_recon_cycle_A = self.criterion_recon(cycle_A, self.real_A)
            loss_recon_cycle_B = self.criterion_recon(cycle_B, self.real_B)
        self.loss_G_GAN = loss_G_GAN_A + loss_G_GAN_B
        self.loss_G_recon = loss_G_recon_A + loss_G_recon_B
        self.loss_G_recon_style = loss_G_recon_s_A + loss_G_recon_s_B
        self.loss_G_recon_content = loss_G_recon_c_A + loss_G_recon_c_B
        # print(self.loss_G_GAN.item(), self.loss_G_recon.item(), self.loss_G_recon_style.item(), self.loss_G_recon_content.item())
        # if self.args.lambda_cycle > 0:
        #     self.loss_G_recon_cycle = loss_recon_cycle_A + loss_recon_cycle_B
        # else:
        #     self.loss_G_recon_cycle = self.loss_G_GAN.fill_(0)
        loss_G = self.loss_G_GAN + self.loss_G_recon + self.loss_G_recon_style + self.loss_G_recon_content # + self.loss_G_recon_cycle
        return loss_G
    def compute_loss(self, x, gt):
        loss = sum([torch.mean((out-gt)**2) for out in x])
        return loss
    def get_loss_D_A(self):
        loss_D_real = self.criterion_GAN(self.D_A(self.real_A), target_is_real=True)
        loss_D_gene = self.criterion_GAN(self.D_A(self.gene_A.detach()), target_is_real=False)
        # loss_D_real = self.compute_loss(self.D_A(self.real_A), 1)
        # loss_D_gene = self.compute_loss(self.D_A(self.gene_A.detach()), 0)
        loss_D_A = loss_D_real + loss_D_gene 
        return loss_D_A
    def get_loss_D_B(self):
        loss_D_real = self.criterion_GAN(self.D_B(self.real_B), target_is_real=True)
        loss_D_gene = self.criterion_GAN(self.D_B(self.gene_B.detach()), target_is_real=False)
        # loss_D_real = self.compute_loss(self.D_B(self.real_B), 1)
        # loss_D_gene = self.compute_loss(self.D_B(self.gene_B.detach()), 0)
        loss_D_B = loss_D_real + loss_D_gene
        return loss_D_B
    def train(self):
        #### 스타일이 도메인이라 생각하자. content는 공유한다고 가정 ####
        #### Generator ####
        self.set_requires_grad([self.E_A, self.E_B, self.G_A, self.G_B], requires_grad=True)
        self.set_requires_grad([self.D_A, self.D_B], requires_grad=False)
        self.loss_G = self.get_loss_G()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()
        
        #### Discriminator A ####
        self.set_requires_grad([self.E_A, self.E_B, self.G_A, self.G_B, self.D_B], requires_grad=False)
        self.set_requires_grad([self.D_A, self.D_B], requires_grad=True)
        self.optimizer_D_A.zero_grad()
        self.loss_D_A = self.get_loss_D_A()
        self.loss_D_A.backward()
        self.optimizer_D_A.step()

        #### Discriminator B ####
        self.optimizer_D_B.zero_grad()
        self.loss_D_B = self.get_loss_D_B()
        self.loss_D_B.backward()
        self.optimizer_D_B.step()

    def to_train(self):
        self.E_A.train()
        self.E_B.train()
        self.G_A.train()
        self.G_B.train()
        self.D_A.train()
        self.D_B.train()
    def to_eval(self):
        self.E_A.eval()
        self.E_B.eval()
        self.G_A.eval()
        self.G_B.eval()
        self.D_A.eval()
        self.D_B.eval()
    def save(self, to_path):
        sd = {}
        if self.args.use_DDP:
            sd["E_A"] = self.E_A.module.state_dict()
            sd["E_B"] = self.E_B.module.state_dict()
            sd["G_A"] = self.G_A.module.state_dict()
            sd["G_B"] = self.G_B.module.state_dict()
            sd["D_A"] = self.D_A.module.state_dict()
            sd["D_B"] = self.D_B.module.state_dict()
        else:
            sd["E_A"] = self.E_A.state_dict()
            sd["E_B"] = self.E_B.state_dict()
            sd["G_A"] = self.G_A.state_dict()
            sd["G_B"] = self.G_B.state_dict()
            sd["D_A"] = self.D_A.state_dict()
            sd["D_B"] = self.D_B.state_dict()
        if self.args.local_rank==0:
            torch.save(sd, to_path)
    def load(self, load_path):
        state_dict = torch.load(load_path)
        self.E_A.load_state_dict(state_dict["E_A"])
        self.E_B.load_state_dict(state_dict["E_B"])
        self.G_A.load_state_dict(state_dict["G_A"])
        self.G_B.load_state_dict(state_dict["G_B"])
        print("models are successfully loaded from {load_path}")
    def synthesize(self, real_A, real_B):
        with torch.no_grad():
            c_A, s_A = self.E_A(real_A)
            c_B, s_B = self.E_B(real_B)
            cA_sB = self.G_B(c_A, s_B)
            cB_sA = self.G_A(c_B, s_A)
            return cA_sB, cB_sA          

        