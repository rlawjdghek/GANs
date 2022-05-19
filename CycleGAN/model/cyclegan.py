import itertools
import os
from os.path import join as opj
import math

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torchvision
from pytorch_msssim import MS_SSIM, SSIM

from .base_model import BaseModel
from .networks import define_G, define_D, get_scheduler, GANLoss
from util.image_pool import ImagePool
from util.utils import AverageMeter, _psnr

class CycleGANModel(BaseModel):
    '''
    A: ncct, B: cect
    '''
    def __init__(self, args, logger):
        BaseModel.__init__(self, args, logger)
        self.G_AB = define_G(G_name=args.G_AB_name,
                            input_ch=args.input_ch,
                            output_ch=args.output_ch,
                            ngf=args.ngf,
                            norm_type=args.G_norm_type,
                            init_type=args.G_init_type,
                            init_gain=args.G_init_gain,
                            use_dropout=args.G_use_dropout).cuda(args.local_rank)  # real A -> gene B -> recon A
        self.G_BA = define_G(G_name=args.G_BA_name,
                            input_ch=args.input_ch,
                            output_ch=args.output_ch,
                            ngf=args.ngf,
                            norm_type=args.G_norm_type,
                            init_type=args.G_init_type,
                            init_gain=args.G_init_gain,
                            use_dropout=args.G_use_dropout).cuda(args.local_rank)  # real B -> gene A -> recon B
        self.D_A = define_D(D_name=args.D_A_name,
                           input_ch=args.input_ch,
                           ndf=args.ndf,
                           n_layers=args.D_n_layers,
                           norm_type=args.D_norm_type,
                           init_type=args.D_init_type,
                           init_gain=args.D_init_gain).cuda(args.local_rank)  # real A와 gene A를 비교
        self.D_B = define_D(D_name=args.D_B_name,
                           input_ch=args.input_ch,
                           ndf=args.ndf,
                           n_layers=args.D_n_layers,
                           norm_type=args.D_norm_type,
                           init_type=args.D_init_type,
                           init_gain=args.D_init_gain).cuda(args.local_rank)  # real B와 gene B를 비교
        if args.DDP:
            self.G_AB = torch.nn.parallel.DistributedDataParallel(self.G_AB, device_ids=[args.local_rank])
            self.G_BA = torch.nn.parallel.DistributedDataParallel(self.G_BA, device_ids=[args.local_rank])
            self.D_A = torch.nn.parallel.DistributedDataParallel(self.D_A, device_ids=[args.local_rank])
            self.D_B = torch.nn.parallel.DistributedDataParallel(self.D_B, device_ids=[args.local_rank])

        self.gene_A_pool = ImagePool(args.pool_size)
        self.gene_B_pool = ImagePool(args.pool_size)
        self.criterion_GAN = GANLoss(args.gan_loss_name, target_real_label=args.target_real_label,
                                     target_gene_label=args.target_gene_label).cuda(self.args.local_rank)
        self.criterion_Cycle = torch.nn.L1Loss()
        self.criterion_ID = torch.nn.L1Loss()
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
                                            lr=args.G_lr, betas=args.G_betas)
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
                                           lr=args.D_lr, betas=args.D_betas)
        self.scheduler_G = get_scheduler(self.args, optimizer=self.optimizer_G)
        self.scheduler_D = get_scheduler(self.args, optimizer=self.optimizer_D)
        self.scaler = GradScaler()

        self.train_loss_D = AverageMeter()
        self.train_loss_G = AverageMeter()
        self.train_msssim_A = AverageMeter()
        self.train_msssim_B = AverageMeter()
        self.train_psnr_A = AverageMeter()
        self.train_psnr_B = AverageMeter()
        self.valid_msssim_A = AverageMeter()
        self.valid_msssim_B = AverageMeter()
        self.valid_psnr_A = AverageMeter()
        self.valid_psnr_B = AverageMeter()
        self.MSSSIM = MS_SSIM(data_range=1)
        self.SSIM = SSIM(data_range=1)
        self.denorm_T = torchvision.transforms.Normalize((-1, -1, -1), (2, 2, 2))
    def to_train(self):
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()
    def to_eval(self):
        self.G_AB.eval()
        self.G_BA.eval()
        self.D_A.eval()
        self.D_B.eval()
    def reset_meters(self):
        self.train_loss_D.reset()
        self.train_loss_G.reset()
        self.train_msssim_A.reset()
        self.train_msssim_B.reset()
        self.valid_msssim_A.reset()
        self.valid_msssim_B.reset()
    def set_input(self, real_A, real_B):
        self.real_A = real_A
        self.real_B = real_B
    def forward_G(self):
        self.gene_B = self.G_AB(self.real_A)
        self.rec_A = self.G_BA(self.gene_B)
        self.gene_A = self.G_BA(self.real_B)
        self.rec_B = self.G_AB(self.gene_A)
    def get_loss_G(self):
        if self.args.lambda_ID > 0:
            self.id_A = self.G_BA(self.real_A)
            self.id_B = self.G_AB(self.real_B)
            self.id_loss_A = self.criterion_ID(self.id_A, self.real_A) * self.args.lambda_A * self.args.lambda_ID
            self.id_loss_B = self.criterion_ID(self.id_B, self.real_B) * self.args.lambda_B * self.args.lambda_ID

        else:
            self.id_loss_A = 0
            self.id_loss_B = 0

        self.loss_GAN_A = self.criterion_GAN(self.D_A(self.gene_A), target_is_real=True)
        self.loss_GAN_B = self.criterion_GAN(self.D_B(self.gene_B), target_is_real=True)
        self.loss_cycle_A = self.criterion_Cycle(self.rec_A, self.real_A) * self.args.lambda_A
        self.loss_cycle_B = self.criterion_Cycle(self.rec_B, self.real_B) * self.args.lambda_B
        loss_G =  self.id_loss_A + self.id_loss_B + self.loss_GAN_A + self.loss_GAN_B + \
                  self.loss_cycle_A + self.loss_cycle_B
        return loss_G
    def get_loss_D(self):
        gene_A = self.gene_A_pool.query(self.gene_A)  # pool size갯수만큼의 이전 생성 이미지에서 뽑아냄
        pred_real = self.D_A(self.real_A)
        pred_gene = self.D_A(gene_A.detach())
        loss_D_real = self.criterion_GAN(pred_real, target_is_real=True)
        loss_D_gene = self.criterion_GAN(pred_gene, target_is_real=False)
        self.loss_D_A = (loss_D_real + loss_D_gene) / 2

        gene_B = self.gene_B_pool.query(self.gene_B)
        pred_real = self.D_B(self.real_B)
        pred_gene = self.D_B(gene_B.detach())
        loss_D_real = self.criterion_GAN(pred_real, target_is_real=True)
        loss_D_gene = self.criterion_GAN(pred_gene, target_is_real=False)
        self.loss_D_B = (loss_D_real + loss_D_gene) / 2
        loss_D = self.loss_D_A + self.loss_D_B
        return loss_D
    def train(self, iter):
        self.set_requires_grad([self.D_A, self.D_B], requires_grad=False)  # 없어도 어차피 get_loss_G함수에서 계산은 안하지만 속
        # 도가 다르다.
        with autocast():
            self.forward_G()
            self.loss_G = self.get_loss_G()
        self.optimizer_G.zero_grad()
        self.scaler.scale(self.loss_G).backward()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()

        self.set_requires_grad([self.D_A, self.D_B], requires_grad=True)  # D 다시 풀어줌
        with autocast():
            self.loss_D = self.get_loss_D()
        self.optimizer_D.zero_grad()
        self.scaler.scale(self.loss_D).backward()
        self.scaler.step(self.optimizer_D)
        self.scaler.update()

        self.train_loss_G.update(self.loss_G.item(), self.real_A.shape[0])
        self.train_loss_D.update(self.loss_D.item(), self.real_A.shape[0])

        if iter % self.args.img_save_iter_freq <= self.args.n_save_images:
            _idx = iter % self.args.img_save_iter_freq
            self.img_save(img_idx=_idx, iter=iter, is_train=True)
        if iter % self.args.model_save_iter_freq == 0:
            to_path = opj(self.args.model_save_dir, f"[iteration-{iter}_{self.args.total_iter}].pth")
            self.model_save(to_path)
    def img_save(self, img_idx, iter=None, epoch=None, is_train=True):
        if is_train:
            to_path = opj(self.args.img_save_dir, "NtoC", f"[train]_[iteration-{iter}_{self.args.total_iter}]_{img_idx}.png")
            self._img_save(self.real_A.detach()[:self.args.n_save_images], self.gene_B.detach()[:self.args.n_save_images], self.real_B.detach()[:self.args.n_save_images], to_path)
            to_path = opj(self.args.img_save_dir, "CtoN", f"[train]_[CtoN]_[iteration-{iter}_{self.args.total_iter}]_{img_idx}.png")
            self._img_save(self.real_B.detach()[:self.args.n_save_images], self.gene_A.detach()[:self.args.n_save_images], self.real_A.detach()[:self.args.n_save_images], to_path)
        else:
            to_path = opj(self.args.img_save_dir, "NtoC", f"[valid]_[epoch-{epoch}_{self.args.n_epochs}]_{img_idx}.png")
            self._img_save(self.real_A.detach()[:self.args.n_save_images], self.gene_B.detach()[:self.args.n_save_images], self.real_B.detach()[:self.args.n_save_images], to_path)
            to_path = opj(self.args.img_save_dir, "CtoN", f"[valid]_[epoch-{epoch}_{self.args.n_epochs}]_{img_idx}.png")
            self._img_save(self.real_B.detach()[:self.args.n_save_images], self.gene_A.detach()[:self.args.n_save_images], self.real_A.detach()[:self.args.n_save_images], to_path)
    def validation(self, epoch, iter):
        with torch.no_grad():
            with autocast():
                self.forward_G()
        self.img_save(img_idx=iter, epoch=epoch, is_train=False)
    def calc_msssim_psnr(self, is_train):
        self.forward_G()
        denorm_real_A = self.denorm_T(self.real_A.detach())
        denorm_real_B = self.denorm_T(self.real_B.detach())
        denorm_gene_A = self.denorm_T(self.gene_A.detach())
        denorm_gene_B = self.denorm_T(self.gene_B.detach())
        n = self.real_A.shape[0]
        ms_A = self.MSSSIM(denorm_real_A, denorm_gene_A)
        ms_B = self.MSSSIM(denorm_real_B, denorm_gene_B)
        psnr_A = _psnr(denorm_real_A, denorm_gene_A)
        psnr_B = _psnr(denorm_real_B, denorm_gene_B)
        if is_train:
            self.train_msssim_A.update(ms_A, n)
            self.train_msssim_B.update(ms_B, n)
            self.train_psnr_A.update(psnr_A, n)
            self.train_psnr_B.update(psnr_B, n)
        else:
            self.valid_msssim_A.update(ms_A, n)
            self.valid_msssim_B.update(ms_B, n)
            self.valid_psnr_A.update(psnr_A, n)
            self.valid_psnr_B.update(psnr_B, n)
    def model_save(self, to_path):
        state_dict = {}
        if self.args.DDP or self.args.DP and self.args.local_rank==0:
            state_dict["G_AB"] = self.G_AB.module.state_dict()
            state_dict["G_BA"] = self.G_BA.module.state_dict()
            state_dict["D_A"] = self.D_A.module.state_dict()
            state_dict["D_B"] = self.D_B.module.state_dict()
        else:
            state_dict["G_AB"] = self.G_AB.state_dict()
            state_dict["G_BA"] = self.G_BA.state_dict()
            state_dict["D_A"] = self.D_A.state_dict()
            state_dict["D_B"] = self.D_B.state_dict()
        state_dict["optimizer_G"] = self.optimizer_G.state_dict()
        state_dict["optimizer_D"] = self.optimizer_D.state_dict()
        torch.save(state_dict, to_path)
    @staticmethod
    def _img_save(input_img, gene_img, real_img, to_path):
        input_img = torchvision.utils.make_grid(input_img, nrow=1, padding=0)
        gene_img = torchvision.utils.make_grid(gene_img, nrow=1, padding=0)
        real_img = torchvision.utils.make_grid(real_img, nrow=1, padding=0)
        input_img = F.pad(input_img, pad=(4, 4, 4, 4))
        gene_img = F.pad(gene_img, pad=(4, 4, 4, 4))
        real_img = F.pad(real_img, pad=(4, 4, 4, 4))
        save_img = torch.cat([input_img, gene_img, real_img], dim=2)
        torchvision.utils.save_image(save_img, to_path, normalize=True)



