import os
from os.path import join as opj
import shutil

import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .base_model import BaseModel, GANLoss, R1RegLoss, moving_average
from .base_network import define_D, define_E, define_G, define_F, count_params
class STARGANv2(BaseModel):
    def __init__(self, args):
        self.args = args
        self.D = define_D(args).cuda(args.local_rank)
        self.G = define_G(args).cuda(args.local_rank)
        self.E = define_E(args).cuda(args.local_rank)
        self.F = define_F(args).cuda(args.local_rank)
        self.n_params_D = count_params(self.D)
        self.n_params_G = count_params(self.G)
        self.n_params_E = count_params(self.E)
        self.n_params_F = count_params(self.F)
        if args.use_DDP:
            self.D = DDP(self.D, device_ids=[args.local_rank])
            self.G = DDP(self.G, device_ids=[args.local_rank])
            self.E = DDP(self.E, device_ids=[args.local_rank])
            self.F = DDP(self.F, device_ids=[args.local_rank])
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=args.D_lr, betas=args.betas, weight_decay=args.weight_decay)
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=args.G_lr, betas=args.betas, weight_decay=args.weight_decay)
        self.optimizer_E = optim.Adam(self.E.parameters(), lr=args.E_lr, betas=args.betas, weight_decay=args.weight_decay)
        self.optimizer_F = optim.Adam(self.F.parameters(), lr=args.F_lr, betas=args.betas, weight_decay=args.weight_decay)

        self.criterion_GAN = GANLoss()
        self.criterion_r1_reg = R1RegLoss()
        self.criterion_L1 = nn.L1Loss()

        self.G_ema = define_G(args).cuda(args.local_rank).eval()
        self.E_ema = define_E(args).cuda(args.local_rank).eval()
        self.F_ema = define_F(args).cuda(args.local_rank).eval()
    def set_input(self, src_img, src_label, ref_img1, ref_img2, ref_label, z, z2):
        self.real_A = src_img
        self.label_A = src_label
        self.real_B = ref_img1
        self.real_B2 = ref_img2
        self.label_B = ref_label
        self.z = z
        self.z2 = z2
    def reset_grad(self):
        self.optimizer_D.zero_grad()
        self.optimizer_G.zero_grad()
        self.optimizer_E.zero_grad()
        self.optimizer_F.zero_grad()
    def train(self):
        # update D
        self.update_D()
        
        # update GEF
        self.update_GEF()
    def update_D(self):
        # style from z
        self.real_A.requires_grad_()
        pred_real = self.D(self.real_A, self.label_A)
        with torch.no_grad():
            style_B = self.F(self.z, self.label_B)
            gene_B = self.G(self.real_A, style_B)
        # pred_gene = self.D(gene_B.detach(), self.label_B)
        pred_gene = self.D(gene_B, self.label_B)
        loss_D_adv_real = self.criterion_GAN(pred_real, is_target_real=True)
        loss_D_adv_gene = self.criterion_GAN(pred_gene, is_target_real=False)
        loss_D_adv = loss_D_adv_real + loss_D_adv_gene
        loss_D_reg = self.criterion_r1_reg(pred_real, self.real_A) * self.args.lambda_reg
        loss_D_z = loss_D_adv + loss_D_reg
        self.reset_grad()
        loss_D_z.backward()
        self.optimizer_D.step()     

        #### loss 기록용 ####
        self.D_latent_real_val = loss_D_adv_real.item()
        self.D_latent_gene_val = loss_D_adv_gene.item()
        self.D_latent_reg_val = loss_D_reg.item()

        # style from ref
        pred_real = self.D(self.real_A, self.label_A)
        with torch.no_grad():
            style_B = self.E(self.real_B, self.label_B)
            gene_B = self.G(self.real_A, style_B)
        # pred_gene = self.D(gene_B.detach(), self.label_B)
        pred_gene = self.D(gene_B, self.label_B)
        loss_D_adv_real = self.criterion_GAN(pred_real, is_target_real=True)
        loss_D_adv_gene = self.criterion_GAN(pred_gene, is_target_real=False)
        loss_D_adv = loss_D_adv_real + loss_D_adv_gene
        loss_D_reg = self.criterion_r1_reg(pred_real, self.real_A) * self.args.lambda_reg
        loss_D_ref = loss_D_adv + loss_D_reg
        self.reset_grad()
        loss_D_ref.backward()
        self.optimizer_D.step()

        #### loss 기록용 ####
        self.loss_D = loss_D_z.detach() + loss_D_ref.detach()
        self.D_ref_real_val = loss_D_adv_real.item()
        self.D_ref_gene_val = loss_D_adv_gene.item()
        self.D_ref_ref_val = loss_D_ref.item()
    def update_GEF(self):
        #### style from z ####
        style_B = self.F(self.z, self.label_B)
        gene_B = self.G(self.real_A, style_B)
        pred_gene = self.D(gene_B, self.label_B)

        # adv loss
        loss_adv = self.criterion_GAN(pred_gene, is_target_real=True)

        # style reconstruction loss
        style_gene_B = self.E(gene_B, self.label_B)
        loss_style = self.criterion_L1(style_gene_B, style_B) * self.args.lambda_style

        # ds loss
        style_B2 = self.F(self.z2, self.label_B)
        gene_B2 = self.G(self.real_A, style_B2).detach()
        loss_ds = self.criterion_L1(gene_B, gene_B2) * self.args.lambda_ds

        # cycle loss
        style_A = self.E(self.real_A, self.label_A)
        cycle_A = self.G(gene_B, style_A)
        loss_cycle = self.criterion_L1(cycle_A, self.real_A) * self.args.lambda_cycle

        loss_G_z = loss_adv + loss_style - loss_ds + loss_cycle
        self.reset_grad()
        loss_G_z.backward()
        self.optimizer_G.step()
        self.optimizer_E.step()
        self.optimizer_F.step()

        #### loss 기록용 ####
        self.G_latent_adv_val = loss_adv.item()
        self.G_latent_sty_val = loss_style.item()
        self.G_latent_ds_val = loss_ds.item()
        self.G_latent_cyc_val = loss_cycle.item()

        #### style from ref ####
        style_B = self.E(self.real_B, self.label_B)
        gene_B = self.G(self.real_A, style_B)
        pred_gene = self.D(gene_B, self.label_B)

        # adv loss 
        loss_adv = self.criterion_GAN(pred_gene, is_target_real=True)
        
        # style reconstruction loss
        style_gene_B = self.E(gene_B, self.label_B)
        loss_style = self.criterion_L1(style_gene_B, style_B) * self.args.lambda_style

        # ds loss 
        style_B2 = self.E(self.real_B2, self.label_B)
        gene_B2 = self.G(self.real_A, style_B2).detach()
        loss_ds = self.criterion_L1(gene_B, gene_B2) * self.args.lambda_ds

        # cycle loss
        style_A = self.E(self.real_A, self.label_A)
        cycle_A = self.G(gene_B, style_A)
        loss_cycle = self.criterion_L1(cycle_A, self.real_A) * self.args.lambda_cycle

        loss_G_ref = loss_adv + loss_style - loss_ds + loss_cycle
        self.reset_grad()
        loss_G_ref.backward()
        self.optimizer_G.step()

        self.loss_G = loss_G_z.detach() + loss_G_ref.detach()
        
        #### loss 기록용 ####
        self.G_ref_adv_val = loss_adv.item()
        self.G_ref_sty_val = loss_style.item()
        self.G_ref_ds_val = loss_ds.item()
        self.G_ref_cyc_val = loss_cycle.item()
    def DDP_save_load(self, save_path):
        if self.args.local_rank == 0:
            state_dict = {}
            state_dict["G"] = self.G.state_dict()
            state_dict["D"] = self.D.state_dict()
            state_dict["E"] = self.E.state_dict()
            state_dict["F"] = self.F.state_dict()
            torch.save(state_dict, save_path)                
        dist.barrier()
        map_loc = {"cuda:0": f"cuda:{self.args.local_rank}"}
        load_state_dict = torch.load(save_path, map_location=map_loc)
        self.G.load_state_dict(load_state_dict["G"])
        self.D.load_state_dict(load_state_dict["D"])
        self.E.load_state_dict(load_state_dict["E"])
        self.F.load_state_dict(load_state_dict["F"])
    def save(self, save_path):
        state_dict = {}
        if self.args.use_DDP:
            state_dict["D"] = self.D.module.state_dict()
            state_dict["G"] = self.G.module.state_dict()
            state_dict["E"] = self.E.module.state_dict()
            state_dict["F"] = self.F.module.state_dict()
        else:
            state_dict["D"] = self.D.state_dict()
            state_dict["G"] = self.G.state_dict()
            state_dict["E"] = self.E.state_dict()
            state_dict["F"] = self.F.state_dict()
        state_dict["G_ema"] = self.G_ema.state_dict()
        state_dict["E_ema"] = self.E_ema.state_dict()
        state_dict["F_ema"] = self.F_ema.state_dict()
        torch.save(state_dict, save_path)
    def update_moving_avg(self):
        if self.args.use_DDP:            
            moving_average(self.G.module, self.G_ema, beta=0.999)
            moving_average(self.E.module, self.E_ema, beta=0.999)
            moving_average(self.F.module, self.F_ema, beta=0.999)
        else:
            moving_average(self.G, self.G_ema, beta=0.999)
            moving_average(self.E, self.E_ema, beta=0.999)
            moving_average(self.F, self.F_ema, beta=0.999)
    @torch.no_grad() 
    def ema_inference(self):
        #### latent-guided ####
        style_B = self.F_ema(self.z, self.label_B)
        gene_B = self.G_ema(self.real_A, style_B)
        style_B2 = self.F_ema(self.z2, self.label_B)
        gene_B2 = self.G_ema(self.real_A, style_B2)
        style_A = self.E_ema(self.real_A, self.label_A)
        cycle_A = self.G_ema(gene_B, style_A)
        
        self.gene_B_latent = gene_B.detach()
        self.gene_B_latent2 = gene_B2.detach()
        self.cycle_A = cycle_A.detach()

        #### reference-guided ####
        style_B = self.E_ema(self.real_B, self.label_B)
        gene_B = self.G_ema(self.real_A, style_B)
        style_B2 = self.E_ema(self.real_B2, self.label_B)
        gene_B2 = self.G_ema(self.real_A, style_B2)
        
        self.gene_B_ref = gene_B.detach()
        self.gene_B_ref2 = gene_B2.detach()       
    @torch.no_grad()
    def normal_interence(self):
        self.F.eval()
        self.G.eval()
        self.E.eval()
        #### latent-guided ####
        style_B = self.F(self.z, self.label_B)
        gene_B = self.G(self.real_A, style_B)
        style_B2 = self.F(self.z2, self.label_B)
        gene_B2 = self.G(self.real_A, style_B2)
        style_A = self.E(self.real_A, self.label_A)
        cycle_A = self.G(gene_B, style_A)
        
        self.gene_B_latent = gene_B.detach()
        self.gene_B_latent2 = gene_B2.detach()
        self.cycle_A = cycle_A.detach()

        #### reference-guided ####
        style_B = self.E(self.real_B, self.label_B)
        gene_B = self.G(self.real_A, style_B)
        style_B2 = self.E(self.real_B2, self.label_B)
        gene_B2 = self.G(self.real_A, style_B2)
        
        self.gene_B_ref = gene_B.detach()
        self.gene_B_ref2 = gene_B2.detach()
        self.F.train()
        self.G.train()
        self.E.train()
    @torch.no_grad()
    def evaluate(self, mode):
        ### LPIPS ####
        assert mode in ["latent", "reference"]
        from datasets.dataloader import get_single_dataloader
        from utils.util import tensor2img
        from metrics.lpips import calculate_lpips_given_images
        domain_names = sorted(os.listdir(opj(self.args.data_root_dir, self.args.data_name, "val")))
        num_domains = len(domain_names)
        print(f"num domains : {num_domains}")
        lpips_dict = dict()
        for ref_idx, ref_domain in enumerate(domain_names):
            if mode == "reference":
                ref_data_dir = opj(self.args.data_root_dir, self.args.data_name, "val", ref_domain)
                ref_loader = get_single_dataloader(data_dir=ref_data_dir, img_size=self.args.img_size, batch_size=self.args.batch_size, imagenet_normalize=False)
            src_domains = [x for x in domain_names if x != ref_domain]
            for src_idx, src_domain in enumerate(src_domains):
                src_data_dir = opj(self.args.data_root_dir, self.args.data_name, "val", src_domain)
                src_loader = get_single_dataloader(data_dir=src_data_dir, img_size=self.args.img_size, batch_size=self.args.batch_size, imagenet_normalize=False)
                task = f"{src_domain}_to_{ref_domain}"
                print(f"Evaluating LPIPS on {task}....")
                save_dir = opj(self.args.eval_save_dir, task)
                shutil.rmtree(save_dir, ignore_errors=True)
                os.makedirs(save_dir)

                lpips_val_lst = []
                for i, src_data in enumerate(src_loader):
                    real_A = src_data["img"].cuda(self.args.local_rank)
                    BS = real_A.shape[0]
                    label_B = torch.tensor([ref_idx]*BS).cuda(self.args.local_rank)
                    
                    group_gene_imgs = []
                    for j in range(self.args.n_outs_per_domain):  # 1개의 src마다 ref 10장씩. loader마다 동일한 도메인임을 명심.
                        if mode == "latent":
                            z = torch.randn(BS, self.args.latent_dim).cuda(self.args.local_rank)
                            style_B = self.F_ema(z, label_B)
                        else:
                            try:
                                real_B = next(iter_ref).cuda(self.args.local_rank)
                            except:
                                iter_ref = iter(ref_loader)
                                real_B = next(iter_ref)["img"].cuda(self.args.local_rank)
                            style_B = self.E_ema(real_B, label_B)
                        gene_B = self.G_ema(real_A, style_B)
                        group_gene_imgs.append(gene_B)
                        for k in range(BS):
                            to_path = opj(save_dir, f"{i*self.args.batch_size + (k+1):02d}_{j+1:02d}.png")
                            gene_B_img = tensor2img(gene_B[k])
                            cv2.imwrite(to_path, gene_B_img[:,:,::-1])
                    tmp_lpips_val = calculate_lpips_given_images(group_gene_imgs, local_rank=self.args.local_rank)
                    lpips_val_lst.append(tmp_lpips_val)
                    break
                lpips_val = np.array(lpips_val_lst).mean()
                lpips_dict[task] = lpips_val

            del src_loader
            if mode == "reference":
                del ref_loader
                del iter_ref
        lpips_mean = 0
        for _, value in lpips_dict.items():
            lpips_mean += value
        lpips_mean /= len(lpips_dict)
        lpips_dict["mean"] = lpips_mean

        #### FID ####
        from metrics.fid import calculate_fid_given_paths
        fid_dict = {}
        for ref_domain in domain_names:
            src_domains = [x for x in domain_names if x != ref_domain]
            for src_domain in src_domains:
                task = f"{src_domain}_to_{ref_domain}"
                print(f"Evaluating FID on {task}....")
                real_dir = opj(self.args.data_root_dir, self.args.data_name, "train", ref_domain)
                gene_dir = opj(self.args.eval_save_dir, task)
                fid_val = calculate_fid_given_paths([real_dir, gene_dir], img_size=self.args.img_size, batch_size=self.args.batch_size)
                fid_dict[task] = fid_val
        fid_mean = 0
        for _, value in fid_dict.items():
            fid_mean += value
        fid_mean /= len(fid_dict)
        fid_dict["mean"] = fid_mean
        
        save_msg = ""
        for k, v in lpips_dict.items():
            save_msg += f"[{mode} lpips {k} - {v:.4f}]"
        for k, v in fid_dict.items():
            save_msg += f"[{mode} fid {k} - {v:.4f}]"
        return lpips_dict, fid_dict, save_msg
    def print_n_params(self, logger):
        logger.write(f"# of D parameters : {self.n_params_D}")
        logger.write(f"# of G parameters : {self.n_params_G}")
        logger.write(f"# of E parameters : {self.n_params_E}")
        logger.write(f"# of F parameters : {self.n_params_F}")           
