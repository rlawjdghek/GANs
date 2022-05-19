import argparse
from datetime import datetime
import time
import os
from os.path import join as opj

import torch
import torch.distributed as dist

from datasets.dataloader import get_dataloader
from utils.util import *
from models.cut_model import CUTModel

def build_args():
    parser = argparse.ArgumentParser()

    #### dataset ####
    parser.add_argument("--data_root_dir", type=str, default="/home/data/CECTGAN")
    parser.add_argument("--data_name", type=str, default="CECT_all")
    parser.add_argument("--input_ch", type=int, default=1)
    parser.add_argument("--output_ch", type=int, default=1)
    parser.add_argument("--resize", type=bool, default=True)
    parser.add_argument("--crop", type=bool, default=True)
    parser.add_argument("--size", type=int, default=572)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--flip", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)

    #### model ####
    parser.add_argument("--nce_layers", type=str, default="0,4,8,12,16")
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--netG", type=str, default="resnet_9blocks")
    parser.add_argument("--netD", type=str, default="basic")
    parser.add_argument("--netF", type=str, default="mlp_sample")
    parser.add_argument("--netF_nc", type=int, default=256)
    parser.add_argument("--normG", type=str, default="instance")
    parser.add_argument("--normD", type=str, default="instance")
    parser.add_argument("--use_dropout", type=bool, default=False)
    parser.add_argument("--init_type", type=str, default="xavier")
    parser.add_argument("--n_layers_D", type=int, default=3)
    parser.add_argument("--gan_mode", type=str, default="lsgan")
    parser.add_argument("--G_lr", type=float, default=2e-4)
    parser.add_argument("--D_lr", type=float, default=2e-4)
    parser.add_argument("--betas", default=(0.5,0.999))

    #### train ####
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_epochs", type=int, default=50000)
    parser.add_argument("--lambda_NCE", type=float, default=1.0)
    parser.add_argument("--lambda_GAN", type=float, default=1.0)
    parser.add_argument("--num_patches", type=int, default=512)

    #### save & load ####
    parser.add_argument("--save_root_dir", type=str, default="/media/data1/jeonghokim/CECT/save/CUT")
    parser.add_argument("--save_name", type=str, default=f"{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--train_img_save_iter_freq", type=int, default=1000)
    parser.add_argument("--train_model_save_iter_freq", type=int, default=1000)
    parser.add_argument("--log_save_iter_freq", type=int, default=500)
    parser.add_argument("--valid_epoch_freq", type=int, default=2)
    parser.add_argument("--n_save_images", type=int, default=8)

    #### config ####
    parser.add_argument("--use_DDP", type=str, default=True)

    args = parser.parse_args()
    if not args.use_DDP: args.local_rank = 0
    else: args.local_rank = int(os.environ["LOCAL_RANK"])
    args.model_save_dir = opj(args.save_root_dir, args.save_name, "save_models")
    args.img_save_dir = opj(args.save_root_dir, args.save_name, "save_images")
    args.logger_path = opj(args.save_root_dir, args.save_name, "log.txt")
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.img_save_dir, exist_ok=True)
    return args

def main_worker(args, logger):
    train_loader, valid_loader = get_dataloader(args, logger)
    args.total_iter = args.n_epochs * len(train_loader)
    model = CUTModel(args)
    logger.write(model.G.__str__())
    logger.write(model.D.__str__())
    cur_iter = 1
    start_epoch = 1
    loss_D_adv_val = AverageMeter()
    loss_G_adv_val = AverageMeter()
    loss_G_NCE_val = AverageMeter()
    for epoch in range(start_epoch, args.n_epochs+1):
        start_time = time.time()
        if args.use_DDP: train_loader.sampler.set_epoch(epoch)
        model.to_train()
        for data in train_loader:
            ncct_img = data["ncct_img"].cuda(args.local_rank)
            cect_img = data["cect_img"].cuda(args.local_rank)

            if args.use_DDP: torch.cuda.synchronize()
            # if epoch == start_epoch and cur_iter == 0:  # F가 input에 따라 달라지므로 여기서 초기화 작업을 진행한다.
            #     model.data_dependent_init(ncct_img, cect_img)
            model.set_input(ncct_img, cect_img)
            model.train()
            bs = args.batch_size
            loss_D_adv_val.update(model.loss_D_adv.item(), bs)
            loss_G_adv_val.update(model.loss_G_adv.item(), bs)
            loss_G_NCE_val.update(model.loss_G_NCE.item(), bs)
            if cur_iter % args.train_img_save_iter_freq == 0:
                ncct_img_cp = ncct_img.detach().cpu()
                cect_img_cp = cect_img.detach().cpu()
                gene_img_cp = model.gene_B.detach().cpu()
                to_path = opj(args.img_save_dir, f"[train]_[iter - {cur_iter}].png")
                img_save(ncct_img_cp, gene_img_cp, cect_img_cp, to_path, local_rank=args.local_rank)
            if cur_iter % args.train_model_save_iter_freq == 0:
                to_path = opj(args.model_save_dir, f"[iter - {cur_iter}].pth")
                model.save(to_path)
            if cur_iter % args.log_save_iter_freq == 0:
                msg = f"[iter - {cur_iter}/{args.total_iter}]_[Time - {time.time() - start_time:.2f}]_[D adv - {loss_D_adv_val.avg:6f}]_[G adv - {loss_G_adv_val.avg:6f}]_[G NCE - {loss_G_NCE_val.avg:6f}]"
                logger.write(msg)
            cur_iter += 1
        if epoch % args.valid_epoch_freq == 0:
            model.to_eval()
            for valid_idx, data in enumerate(valid_loader):
                if valid_idx % 40 != 0: continue
                ncct_img = data["ncct_img"].cuda(args.local_rank)
                cect_img = data["cect_img"].cuda(args.local_rank)
                gene_img = model.inference(ncct_img)
                
                ncct_img_cp = ncct_img.detach().cpu()
                cect_img_cp = cect_img.detach().cpu()
                gene_img_cp = gene_img.detach().cpu()
                to_path = opj(args.img_save_dir, f"[valid]_[epoch - {epoch}].png")
                img_save(ncct_img_cp, gene_img_cp, cect_img_cp, to_path, local_rank=args.local_rank)

        msg = f"[Epoch - {epoch}]_[Time - {time.time() - start_time:.2f}]_[D adv - {loss_D_adv_val.avg:6f}]_[G adv - {loss_G_adv_val.avg:6f}]_[G NCE - {loss_G_NCE_val.avg:6f}]"
        logger.write(msg)

        dist.barrier()

if __name__=="__main__":
    args = build_args()
    logger = Logger(args.local_rank)
    if args.local_rank == 0: logger.open(args.logger_path)
    if args.use_DDP:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
    main_worker(args, logger)

    
