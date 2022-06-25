import os
from os.path import join as opj
import argparse
from datetime import datetime
import time 

import numpy as np
import cv2
import torch
import torch.distributed as dist
from torch.backends import cudnn

from utils.util import *
from datasets.dataloader import get_dataloader
from models.STARGANv2 import STARGANv2
def build_args(is_test=False):
    parser = argparse.ArgumentParser()

    #### dataset ####
    parser.add_argument("--data_root_dir", type=str, default="/home/data")
    parser.add_argument("--data_name", type=str, default="AFHQ")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--in_ch", type=int, default=3)

    #### model ####
    parser.add_argument("--style_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--n_domains", type=int, default=3)
    parser.add_argument("--max_ndf", type=int, default=512)
    parser.add_argument("--max_ngf", type=int, default=512)
    parser.add_argument("--max_nef", type=int, default=512)
    parser.add_argument("--w_hpf", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--D_lr", type=float, default=1e-4)
    parser.add_argument("--G_lr", type=float, default=1e-4)
    parser.add_argument("--F_lr", type=float, default=1e-6)
    parser.add_argument("--E_lr", type=float, default=1e-4)
    parser.add_argument("--betas", default=(0.0, 0.99))
    parser.add_argument("--init_type", type=str, default="normal")

    #### train & eval ####
    parser.add_argument("--total_iter", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--lambda_style", type=float, default=1.0)
    parser.add_argument("--lambda_ds", type=float, default=2.0)  # celebA는 1
    parser.add_argument("--lambda_cycle", type=float, default=1.0)
    parser.add_argument("--ds_iter", type=int, default=100000)
    parser.add_argument("--n_outs_per_domain", type=int, default=10)
    
    #### save ####
    parser.add_argument("--no_save", type=bool, default=False)
    parser.add_argument("--save_root_dir", type=str, default="/media/data1/jeonghokim/GANs/StarGANv2")
    parser.add_argument("--save_name", type=str, default=f"{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--log_save_iter_freq", type=int, default=10)
    parser.add_argument("--img_save_iter_freq", type=int, default=100)
    parser.add_argument("--model_save_iter_freq", type=int, default=5000)
    parser.add_argument("--eval_iter_freq", type=int, default=20000)
    parser.add_argument("--n_save_images", type=int, default=8)

    #### config ####
    parser.add_argument("--use_DDP", type=bool, default=False)

    args = parser.parse_args()
    if is_test:
        args.use_DDP = False
        args.is_test = True
        args.no_save = True
    if args.use_DDP: args.local_rank = int(os.environ["LOCAL_RANK"])
    else: args.local_rank = 0
    args.initial_lambda_ds = args.lambda_ds
    args.save_dir = opj(args.save_root_dir, args.save_name)
    args.img_save_dir = opj(args.save_dir, "save_images")
    args.model_save_dir = opj(args.save_dir, "save_models")
    args.eval_save_dir = opj(args.save_dir, "eval_save_images")
    args.log_path = opj(args.save_dir, "log.txt")
    args.config_path = opj(args.save_dir, "config.json")
    if not args.no_save:
        os.makedirs(args.img_save_dir, exist_ok=True)
        os.makedirs(args.model_save_dir, exist_ok=True)
        os.makedirs(args.eval_save_dir, exist_ok=True)
    return args

def main_worker(args, logger):
    src_loader, ref_loader = get_dataloader(args)
    logger.write(f"[Train] # of imgs : {src_loader.dataset.n_imgs}")
    logger.write(f"1 epochs : {len(src_loader)} iters")
    args.epoch_iter = len(src_loader)
    args.n_epochs = int(args.total_iter // len(src_loader)) + 1
    model = STARGANv2(args)
    model.print_n_params(logger)
    cur_iter = 1
    start_time = time.time()
    for epoch in range(1, args.n_epochs+1):
        loss_D_meter = AverageMeter()
        loss_G_meter = AverageMeter()
        for src_data, ref_data in zip(src_loader, ref_loader):
            src_img = src_data['img'].cuda(args.local_rank)
            src_label = src_data['label'].cuda(args.local_rank)
            ref_img1 = ref_data['img1'].cuda(args.local_rank)
            ref_img2 = ref_data["img2"].cuda(args.local_rank)
            ref_label = ref_data["label"].cuda(args.local_rank)
            BS = src_img.shape[0]
            z1 = torch.randn((BS, args.latent_dim)).cuda(args.local_rank)
            z2 = torch.randn((BS, args.latent_dim)).cuda(args.local_rank)
            model.set_input(
                src_img=src_img,
                src_label=src_label,
                ref_img1=ref_img1,
                ref_img2=ref_img2,
                ref_label=ref_label,
                z=z1,
                z2=z2
            )
            model.train()
            model.update_moving_avg()
            loss_D_meter.update(model.loss_D.item(), BS)
            loss_G_meter.update(model.loss_G.item(), BS)
            if args.lambda_ds > 0:
                args.lambda_ds -= (args.initial_lambda_ds / args.ds_iter)

            if cur_iter % args.log_save_iter_freq == 0:
                msg = f"[Train]_[iter - {cur_iter}/{args.total_iter}]_[Time - {time.time() - start_time:.2f}s]_[D/latent_real - {model.D_latent_real_val:.4f}]_[D/latent_fake - {model.D_latent_gene_val:.4f}]_[D/latent_reg - {model.D_latent_reg_val:.4f}]_[D/ref_real - {model.D_ref_real_val:.4f}]_[D/ref_fake - {model.D_ref_gene_val:.4f}]_[D/ref_reg - {model.D_ref_ref_val:.4f}]_[G/latent_adv - {model.G_latent_adv_val:.4f}]_[G/latent_sty - {model.G_latent_sty_val:.4f}]_[G/latent_ds - {model.G_latent_ds_val:.4f}]_[G/latent_cyc - {model.G_latent_cyc_val:.4f}]_[G/ref_adv - {model.G_ref_adv_val:.4f}]_[G/ref_sty - {model.G_ref_sty_val:.4f}]_[G/ref_ds - {model.G_ref_ds_val:.4f}]_[G/ref_cyc - {model.G_ref_cyc_val:.4f}]_[D/sum - {loss_D_meter.avg:.4f}]_[G/sum - {loss_G_meter.avg:.4f}]"
                # msg = f"[Train]_[iter - {cur_iter}/{args.total_iter}]_[loss D - {loss_D_meter.avg:.4f}]_[loss G - {loss_G_meter.avg:.4f}]_[lambda ds - {args.lambda_ds:.4f}]"
                logger.write(msg)
                logger.write("="*30)
            if cur_iter % args.model_save_iter_freq == 0:
                to_path = opj(args.model_save_dir, f"{cur_iter}_{args.total_iter}.pth")
                model.save(to_path)
            if cur_iter % args.img_save_iter_freq == 0:
                model.ema_inference()
                real_A_img = tensor2img(model.real_A)
                real_B_img = tensor2img(model.real_B)
                real_B2_img = tensor2img(model.real_B2)
                gene_B_latent_img = tensor2img(model.gene_B_latent)
                gene_B_latent2_img = tensor2img(model.gene_B_latent2)
                gene_B_ref_img = tensor2img(model.gene_B_ref)
                gene_B_ref2_img = tensor2img(model.gene_B_ref2)
                cycle_A_img = tensor2img(model.cycle_A)
                save_img = np.concatenate([real_A_img, real_B_img, real_B2_img, gene_B_latent_img, gene_B_latent2_img, gene_B_ref_img, gene_B_ref2_img, cycle_A_img], axis=1)
                to_path = opj(args.img_save_dir, f"{cur_iter}_{args.total_iter}.png")
                if args.local_rank == 0:
                    cv2.imwrite(to_path, save_img[:,:,::-1])
            if cur_iter % args.eval_iter_freq == 0 and args.local_rank == 0:
                # latent-guided evaluation 
                latent_lpips_dict, latent_fid_dict, latent_msg = model.evaluate(mode="latent")
                # reference-guided evaluation
                ref_lpips_dict, ref_fid_dict, ref_msg = model.evaluate(mode="reference")
                msg = f"[Valid]_[iter - {cur_iter}/{args.total_iter}]_"
                msg += latent_msg
                msg += ref_msg
                logger.write(msg)
            if args.use_DDP:
                dist.barrier()            
            cur_iter += 1
        # DDP synchronized
        cp_save_path = opj(args.model_save_dir, "checkpoint.pth")
        if args.use_DDP: 
            model.DDP_save_load(cp_save_path)
        
            
if __name__ == "__main__":
    args = build_args()
    logger = Logger(args.local_rank)
    logger.open(args.log_path)
    print_args(args, logger)
    if args.use_DDP:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
    cudnn.benchmark = True
    main_worker(args, logger)

