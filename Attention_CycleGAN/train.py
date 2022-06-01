import os
from os.path import join as opj
import argparse
import time
from datetime import datetime

import cv2
import torch
import torch.distributed as dist

from datasets.dataloader import get_dataloader, get_test_dataloader
from utils.util import *
from models.cyclegan import AttentionCycleGAN


def build_args(is_test=False):
    parser = argparse.ArgumentParser()

    #### dataset ####
    parser.add_argument("--data_root_dir", type=str, default="/home/data/")
    parser.add_argument("--data_name", type=str, default="horse2zebra")
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--out_ch", type=int, default=3)
    parser.add_argument("--use_crop_A", type=bool, default=False)
    parser.add_argument("--use_crop_B", type=bool, default=False)
    parser.add_argument("--resize_type", type=str, default="scale_height", choices=["resize", "scale_height"])
    parser.add_argument("--resize_H", type=int, default=1080, help="scale 또는 crop있을 때 resize할때 사이즈")
    parser.add_argument("--resize_W", type=int, default=1920)
    parser.add_argument("--img_H", type=int, default=512)
    parser.add_argument("--img_W", type=int, default=512, help="crop할때 이미지 사이즈. crop안하면 이거로 resize")
    
    #### model ####
    parser.add_argument("--G_attn_A_name", type=str, default="basic_attn")
    parser.add_argument("--G_attn_B_name", type=str, default="basic_attn")
    parser.add_argument("--G_AB_name", type=str, default="res_9blks")
    parser.add_argument("--G_BA_name", type=str, default="res_9blks")
    parser.add_argument("--D_AB_name", type=str, default="basic")
    parser.add_argument("--D_BA_name", type=str, default="basic")

    #### train ####
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=10000)
    parser.add_argument("--linearlr_epochs", type=int, default=50, help="linear decay ratio for linear lr scheduler")
    parser.add_argument("--target_real_label", type=float, default=1.0)
    parser.add_argument("--target_gene_label", type=float, default=0.0)
    parser.add_argument("--G_lr", type=float, default=2e-4)
    parser.add_argument("--D_lr", type=float, default=2e-4)
    parser.add_argument("--G_betas", type=tuple, default=(0.5, 0.999))
    parser.add_argument("--D_betas", type=tuple, default=(0.5, 0.999))
    parser.add_argument("--gan_loss_name", type=str, default="lsgan", choices=["lsgan", "wgangp", "vanilla"])
    parser.add_argument("--lr_scheduler", type=str, default="linear", choices=["linear", "step", "plateau", "cosine"])
    parser.add_argument("--lambda_ID", type=float, default=0.5)
    parser.add_argument("--lambda_A", type=float, default=10.0)
    parser.add_argument("--lambda_B", type=float, default=10.0)
    parser.add_argument("--pool_size", type=int, default=50)
    parser.add_argument("--no_vgg", action="store_true")
    parser.add_argument("--attn_thres", type=float, default=0.1)
    parser.add_argument("--use_mask_for_D", type=bool, default=False, help="True이면 논문의 equation 7을 사용한다. 즉, D에 들어갈때 mask를 적용해서 들어간다. 그런데 thresh는 적용이 안됨.") 
    parser.add_argument("--stop_attn_learning_epoch", type=int, default=30, help="이 에폭이후로 attn은 학습 안됨.")

    #### save ####
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--save_root_dir", type=str, default="/media/data1/jeonghokim/VFP290K_GAN/save/cyclegan_attention")
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
        os.makedirs(opj(args.img_save_dir, "A2B"), exist_ok=True)
        os.makedirs(opj(args.img_save_dir, "B2A"), exist_ok=True)
    return args

def main_worker(args, logger):
    train_loader, valid_loader = get_dataloader(args)
    args.total_iter = args.n_epochs * len(train_loader)
    logger.write(f"[Train] # of imgs A : {train_loader.dataset.n_A}, # of imgs B : {train_loader.dataset.n_B}")
    logger.write(f"[Valid] # of imgs A : {valid_loader.dataset.n_A}, # of imgs B : {valid_loader.dataset.n_B}")
    logger.write(f"1 epoch = {len(train_loader)} iters")
    model = AttentionCycleGAN(args)
    cur_iter = 1
    start_time = time.time()
    for epoch in range(args.start_epoch, args.n_epochs+1):
        loss_G_meter = AverageMeter()
        loss_D_meter = AverageMeter()
        for data in train_loader:
            img_A = data['img_A'].cuda(args.local_rank)
            img_B = data["img_B"].cuda(args.local_rank)
            model.set_input(img_A, img_B)
            model.train(epoch)

            BS = img_A.shape[0]
            loss_G_meter.update(model.loss_G.item(), BS)
            loss_D_meter.update(model.loss_D.item(), BS)
            if cur_iter % args.log_save_iter_freq == 0:
                msg = f"[iter - {cur_iter}/{args.total_iter}]_[time - {time.time() - start_time:.2f}sec]_[loss G - {loss_G_meter.avg:.4f}]_[loss D - {loss_D_meter.avg:.4f}]"
                logger.write(msg)
            if cur_iter % args.img_save_iter_freq <= args.n_save_images:
                real_A_img = tensor2img(img_A)
                real_B_img = tensor2img(img_B)
                gene_A_img = tensor2img(model.gene_A)
                gene_B_img = tensor2img(model.gene_B)
                attn_A_img = tensor2img(model.attn_A_viz)
                attn_B_img = tensor2img(model.attn_B_viz)
                A2B_to_path = opj(args.img_save_dir, "A2B", f"{cur_iter}_{cur_iter % args.img_save_iter_freq}.png")
                A2B_save_img = np.concatenate([real_A_img, real_B_img, gene_B_img, attn_A_img], axis=1)
                if args.local_rank == 0:
                    cv2.imwrite(A2B_to_path, A2B_save_img[:,:,::-1])
            
                B2A_to_path = opj(args.img_save_dir, "B2A", f"{cur_iter}_{cur_iter % args.img_save_iter_freq}.png")
                B2A_save_img = np.concatenate([real_B_img, real_A_img, gene_A_img, attn_B_img], axis=1)
                if args.local_rank == 0:
                    cv2.imwrite(B2A_to_path, B2A_save_img[:,:,::-1])
                
            if cur_iter % args.model_save_iter_freq == 0:
                to_path = opj(args.model_save_dir, f"[iter - {cur_iter}].pth")
                model.save(to_path)
            cur_iter += 1
        model.scheduler_G.step()
        model.scheduler_D.step()
        G_lr_val = get_lr(model.optimizer_G)
        D_lr_val = get_lr(model.optimizer_D)
        msg = f"[Epoch - {epoch}/{args.n_epochs}]_[time - {time.time() - start_time:.2f}sec]_[loss G - {loss_G_meter.avg:.4f}]_[loss D - {loss_D_meter.avg:.4f}]_[G lr - {G_lr_val}]_[D lr - {D_lr_val}]"
        logger.write(msg)
        

if __name__ == "__main__":
    args = build_args()
    logger = Logger(args.local_rank)
    logger.open("asd.txt")
    print_args(args, logger)
    save_args(args, args.config_path)
    if args.use_DDP:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
    main_worker(args, logger)
        
