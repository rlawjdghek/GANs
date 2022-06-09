import os
from os.path import join as opj
import argparse
import time
from datetime import datetime

import random
import torch
import numpy as np

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
seed_everything(42)




import cv2
import torch
import torch.distributed as dist
torch.autograd.set_detect_anomaly(True)

from datasets.dataloader import get_dataloader, get_test_dataloader
from utils.util import *
from models.DRIT_pp import DRIT_pp

def build_args(is_test=False):
    parser = argparse.ArgumentParser()

    #### dataset ####
    parser.add_argument("--data_root_dir", type=str, default="/home/data/")
    parser.add_argument("--data_name", type=str, default="selfie2anime")
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--in_ch_A", type=int, default=3)
    parser.add_argument("--in_ch_B", type=int, default=3)
    parser.add_argument("--out_ch", type=int, default=3)
    parser.add_argument("--out_ch_A", type=int, default=3)
    parser.add_argument("--out_ch_B", type=int, default=3)
    parser.add_argument("--use_crop_A", type=bool, default=False)
    parser.add_argument("--use_crop_B", type=bool, default=False)
    parser.add_argument("--resize_type", type=str, default="scale_height", choices=["resize", "scale_height"])
    parser.add_argument("--resize_H", type=int, default=256, help="scale 또는 crop있을 때 resize할때 사이즈")
    parser.add_argument("--resize_W", type=int, default=256)
    parser.add_argument("--img_H", type=int, default=216)
    parser.add_argument("--img_W", type=int, default=216, help="crop할때 이미지 사이즈. crop안하면 이거로 resize")

    #### model ####
    parser.add_argument("--nef_content", type=int, default=64)
    parser.add_argument("--nef_style", type=int, default=64)
    parser.add_argument("--style_dim", type=int, default=8)
    parser.add_argument("--ngf", type=int, default=256)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--ndf_content", type=int, default=256)
    parser.add_argument("--n_D", type=int, default=3)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--D_name", type=str, default="multi_scale")
    parser.add_argument("--G_name", type=str, default="res_4blks")
    
    #### train ####
    parser.add_argument("--batch_size", type=int, default=2, help="반은 real파트, 반은 random 파트")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=1200)
    parser.add_argument("--D_lr", type=float, default=1e-4)
    parser.add_argument("--E_lr", type=float, default=1e-4)
    parser.add_argument("--G_lr", type=float, default=1e-4)
    parser.add_argument("--betas", default=(0.5, 0.999))
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--linearlr_epochs", type=int, default=600)
    parser.add_argument("--lr_scheduler", type=str, default="linear")
    parser.add_argument("--lambda_l2reg", type=float, default=0.01)
    parser.add_argument("--lambda_recon", type=float, default=10.0)
    parser.add_argument("--lambda_cycle", type=float, default=10.0)
    parser.add_argument("--lambda_z_L1", type=float, default=10.0)
    parser.add_argument("--D_c_iter", type=int, default=3, help="D content가 몇배 더 많이 학습됨.")

    #### save ####
    parser.add_argument("--no_save", type=bool, default=False)
    parser.add_argument("--save_root_dir", type=str, default="/media/data1/jeonghokim/GANs/DRIT_pp")
    parser.add_argument("--save_name", type=str, default=f"{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--log_save_iter_freq", type=int, default=100)
    parser.add_argument("--img_save_iter_freq", type=int, default=100)
    parser.add_argument("--model_save_iter_freq", type=int, default=500)
    parser.add_argument("--n_save_images", type=int, default=8)

    #### config ####
    parser.add_argument("--use_DDP", type=bool, default=False)

    args = parser.parse_args()
    args.is_test = is_test
    if is_test:
        args.use_DDP = False
        args.no_save = True
    if args.use_DDP: args.local_rank = int(os.environ["LOCAL_RANK"])
    else: args.local_rank = 0
    args.save_dir = opj(args.save_root_dir, args.save_name)
    args.img_save_dir = opj(args.save_dir, "save_images")
    args.model_save_dir = opj(args.save_dir, "save_models")
    args.log_path = opj(args.save_dir, "log.txt")
    args.config_path = opj(args.save_dir, "config.json")
    if not args.no_save:
        os.makedirs(args.img_save_dir, exist_ok=True)
        os.makedirs(args.model_save_dir, exist_ok=True)
    return args
def main_worker(args, logger):
    train_loader, valid_loader = get_dataloader(args)
    args.total_iter = args.n_epochs * len(train_loader)
    logger.write(f"[Train] # of imgs A : {train_loader.dataset.n_A}, # of imgs B : {train_loader.dataset.n_B}")
    logger.write(f"[Valid] # of imgs A : {valid_loader.dataset.n_A}, # of imgs B : {valid_loader.dataset.n_B}")
    logger.write(f"1 epoch = {len(train_loader)} iters")
    args.epoch_iter = len(train_loader)
    model = DRIT_pp(args)
    cur_iter = 1
    for epoch in range(args.start_epoch, args.n_epochs+1):
        loss_D_meter = AverageMeter()
        loss_G_meter = AverageMeter()
        loss_D_c_meter = AverageMeter()
        loss_G_random_meter = AverageMeter()
        for data in train_loader:
            img_A = data["img_A"].cuda(args.local_rank)
            img_B = data["img_B"].cuda(args.local_rank)
            z1 = torch.randn((img_A.shape[0]//2, args.style_dim)).cuda(args.local_rank)
            z2 = torch.randn((img_A.shape[0]//2, args.style_dim)).cuda(args.local_rank)
            model.set_input(img_A, img_B, z1, z2)
            model.train(cur_iter)
            
            BS = img_A.shape[0] // 2
            try:
                loss_D_meter.update(model.loss_D_A1.item() + model.loss_D_A2.item() + model.loss_D_B1.item() + model.loss_D_B2.item(), BS)
                loss_G_meter.update(model.loss_G.item())
                loss_D_c_meter.update(model.loss_D_c.item(), BS)
                loss_G_random_meter.update(model.loss_G_random.item(), BS)
            except: 
                pass
            
            if cur_iter % args.log_save_iter_freq == 0:
                msg = f"[iter - {cur_iter}/{args.total_iter}]_[loss D - {loss_D_meter.avg:.4f}]_[loss D c - {loss_D_c_meter.avg:.4f}]_[loss G - {loss_G_meter.avg:.4f}]_[loss G random - {loss_G_random_meter.avg:.4f}]"
                logger.write(msg)
            if cur_iter % args.img_save_iter_freq == 0:
                real_A_img = tensor2img(model.real_A)
                real_B_img = tensor2img(model.real_B)
                gene_A_img = tensor2img(model.gene_A)
                gene_B_img = tensor2img(model.gene_B)
                recon_A_img = tensor2img(model.recon_A)
                recon_B_img = tensor2img(model.recon_B)
                gene_random_A1_img = tensor2img(model.gene_random_A1)
                gene_random_A2_img = tensor2img(model.gene_random_A2)
                gene_random_B1_img = tensor2img(model.gene_random_B1)
                gene_random_B2_img = tensor2img(model.gene_random_B2)
                save_img_A = np.concatenate([real_A_img, gene_A_img, recon_A_img, gene_random_A1_img, gene_random_A2_img], axis=1)
                save_img_B = np.concatenate([real_B_img, gene_B_img, recon_B_img, gene_random_B1_img, gene_random_B2_img], axis=1)
                save_img = np.concatenate([save_img_A, save_img_B], axis=0)
                to_path = opj(args.img_save_dir, f"{cur_iter}_{args.total_iter}.png")
                if args.local_rank == 0:
                    cv2.imwrite(to_path, save_img[:,:,::-1])
            cur_iter += 1
            
            
        

if __name__ == '__main__':
    args = build_args()
    logger = Logger(args.local_rank, args.no_save)
    logger.open(args.log_path)
    print_args(args, logger)
    if args.use_DDP:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
    main_worker(args, logger)