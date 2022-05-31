import argparse
import os
from os.path import join as opj
from datetime import datetime
import time

import numpy as np
import cv2
import torch
import torch.distributed as dist
from torch.autograd import Variable

from utils.util import *
from datasets.dataloader import get_dataloader
from models.MUNIT import MUNIT

def build_args(is_test=False):
    parser = argparse.ArgumentParser()
    #### dataset ####
    parser.add_argument("--data_root_dir", type=str, default="/home/data/")
    parser.add_argument("--data_name", type=str, default="selfie2anime")
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--img_H", type=int, default=256)
    parser.add_argument("--img_W", type=int, default=256)

    #### train ####
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--n_epochs", type=int ,default=10000)
    parser.add_argument("--lambda_GAN", type=float, default=1)
    parser.add_argument("--lambda_recon", type=float, default=10)
    parser.add_argument("--lambda_style", type=float, default=1)
    parser.add_argument("--lambda_content", type=float, default=1)
    parser.add_argument("--lambda_cycle", type=float, default=0)
    parser.add_argument("--G_lr", type=float, default=1e-4)
    parser.add_argument("--G_betas", default=(0.5, 0.999))
    parser.add_argument("--D_lr", type=float, default=1e-4)
    parser.add_argument("--D_betas", default=(0.5, 0.999))

    #### model ####
    parser.add_argument("--E_A_name", type=str, default="basic")  # encoder
    parser.add_argument("--E_B_name", type=str, default="basic")
    parser.add_argument("--G_A_name", type=str, default="basic")  # decoder
    parser.add_argument("--G_B_name", type=str, default="basic")
    parser.add_argument("--D_A_name", type=str, default="multi_scale")
    parser.add_argument("--D_B_name", type=str, default="multi_scale")
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--style_dim", type=int, default=8)
    parser.add_argument("--n_downsample", type=int, default=2)
    parser.add_argument("--n_upsample", type=int, default=2)
    parser.add_argument("--n_blks", type=int, default=3)    
    parser.add_argument("--D_n_layers", type=int, default=3)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--n_D", type=int, default=3)
    parser.add_argument("--D_norm_type", type=str, default="in")

    #### save ####
    parser.add_argument("--no_save", type=bool, default=False)
    parser.add_argument("--save_root_dir", type=str, default="/data/jeonghokim/GANs/MUNIT")
    parser.add_argument("--save_name", type=str, default=f"{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--log_save_iter_freq", type=int, default=100)
    parser.add_argument("--img_save_iter_freq", type=int, default=100)
    parser.add_argument("--model_save_iter_freq", type=int, default=999999)
    parser.add_argument("--n_save_images", type=int, default=8)

    #### config ####
    parser.add_argument("--use_DDP", type=bool, default=False)

    args = parser.parse_args()
    args.is_test = is_test
    if is_test:
        args.no_save = True
        args.use_DDP = False
    if args.use_DDP: args.local_rank = int(os.environ["LOCAL_RANK"])
    else: args.local_rank = 0
    args.save_dir = opj(args.save_root_dir, args.save_name)
    args.img_save_dir = opj(args.save_dir, "save_images")
    args.model_save_dir = opj(args.save_dir, "save_models")
    args.log_path = opj(args.save_dir, "log.txt")
    args.config_path = opj(args.save_dir, "config.json")
    if not args.no_save:
        os.makedirs(args.model_save_dir, exist_ok=True)
        os.makedirs(args.img_save_dir, exist_ok=True)
    return args

def main_worker(args, logger):
    train_loader, test_loader = get_dataloader(args)
    args.total_iter = args.n_epochs * len(train_loader)
    model = MUNIT(args)
    cur_iter = 1
    start_time = time.time()
    model.to_train()
    for epoch in range(1, args.n_epochs+1):
        loss_G_meter = AverageMeter()
        loss_D_meter = AverageMeter()
        for data in train_loader:
            img_A = data["img_A"].cuda(args.local_rank)
            img_B = data["img_B"].cuda(args.local_rank)
            BS = img_A.shape[0]
            z_A = Variable(torch.randn((BS, args.style_dim, 1, 1)).cuda(args.local_rank))
            z_B = Variable(torch.randn((BS, args.style_dim, 1 ,1)).cuda(args.local_rank))
            model.set_input(img_A, img_B, z_A, z_B)
            model.train()

            loss_G_meter.update(model.loss_G.item(), BS)
            loss_D_meter.update(model.loss_D_A.item()+model.loss_D_B.item(), BS)

            if cur_iter % args.log_save_iter_freq == 0:
                msg = f"[iter - {cur_iter}/{args.total_iter}_[time - {time.time()-start_time:.2f}]_[loss G - {loss_G_meter.avg:.4f}]_[loss D - {loss_D_meter.avg:.4f}]"
                logger.write(msg)
            if cur_iter % args.img_save_iter_freq == 0:
                model.to_eval()
                cA_sB, cB_sA = model.synthesize(img_A, img_B)
                img_A_img = tensor2img(img_A)
                img_B_img = tensor2img(img_B)
                gene_A_img = tensor2img(model.gene_A)
                gene_B_img = tensor2img(model.gene_B)
                cA_sB_img = tensor2img(cA_sB)
                cB_sA_img = tensor2img(cB_sA)
                save_img = np.concatenate([img_A_img, img_B_img, gene_A_img, gene_B_img, cA_sB_img, cB_sA_img], axis=1)
                to_path = opj(args.img_save_dir, f"{cur_iter}-{args.total_iter}.png")
                if args.local_rank == 0:
                    cv2.imwrite(to_path, save_img[:,:,::-1])
                model.to_train()
            if cur_iter % args.model_save_iter_freq == 0:
                to_path = opj(args.model_save_dir, f"[iter]_{cur_iter}-{args.total_iter}.pth")
                model.save(to_path)
            
            cur_iter += 1        
        to_path = opj(args.model_save_dir, f"[epoch]{epoch}-{args.n_epochs}.pth")
        model.save(to_path)
    
    



if __name__ == "__main__":
    args = build_args()
    logger = Logger(args.local_rank)
    logger.open(args.log_path)
    print_args(args, logger)
    save_args(args, args.config_path)
    if args.use_DDP:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
    main_worker(args, logger)
