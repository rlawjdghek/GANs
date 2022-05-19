import argparse
import os
from os.path import join as opj
from datetime import datetime
import time

import torch
import torch.distributed as dist

from datasets.dataloader import get_dataloader
from utils.utils import *
from models.styleswin import StyleSwinModel

import torch
import numpy as np
import torch.backends.cudnn as cudnn
import random
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)


def build_args():
    parser = argparse.ArgumentParser()
    #### dataset ####
    parser.add_argument("--data_root_dir", type=str, default="/home/data")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=1024)

    #### train ####
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_iters", type=int, default=200000)
    parser.add_argument("--G_lr", type=float, default=2e-4)
    parser.add_argument("--D_lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--G_reg_every", type=int, default=1e+8)
    parser.add_argument("--D_reg_every", type=int, default=16)
    parser.add_argument("--G_ema_decay", type=float, default=0.5**(32/10000))
    parser.add_argument("--gan_weight", type=float, default=1)
    parser.add_argument("--r1", type=float, default=10)
    
    #### model ####
    parser.add_argument("--n_mapping_networks", type=int, default=8)
    parser.add_argument("--style_dim", type=int, default=512)
    parser.add_argument("--layer_init_type", type=str, default="xavier_uniform")

    #### save & load ####
    parser.add_argument("--save_root_dir", type=str, default=f"/media/data1/jeonghokim/GANs/StyleSwin/save/{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--log_save_iter_freq", type=int, default=100)
    parser.add_argument("--img_save_iter_freq", type=int, default=1000)
    parser.add_argument("--model_save_iter_freq", type=int, default=10000000)
    #### config ####
    parser.add_argument("--use_DDP", type=bool, default=False)
    args = parser.parse_args()
    if not args.use_DDP: args.local_rank = 0
    else: args.local_rank = int(os.environ["LOCAL_RANK"])
    args.log_path = opj(args.save_root_dir, "log.txt")
    args.img_save_dir = opj(args.save_root_dir, "save_images")
    args.model_save_dir = opj(args.save_root_dir, "save_models")
    os.makedirs(args.img_save_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    return args

def main_worker(args, logger):
    train_loader, train_sampler = get_dataloader(args)
    model = StyleSwinModel(args, logger)
    n_epochs = args.n_iters // len(train_loader)
    start_time = time.time()
    cur_iter = 1
    for epoch in range(n_epochs+1):
        if args.use_DDP: train_sampler.set_epoch(epoch)
        for img in train_loader:
            img = img[:args.batch_size].cuda(args.local_rank)
            img2 = img[args.batch_size:].cuda(args.local_rank)
            z = torch.randn((args.batch_size, args.style_dim)).cuda(args.local_rank)
            z2 = torch.randn((args.batch_size, args.style_dim)).cuda(args.local_rank)
            model.set_input(img, img2, z, z2)
            model.train(cur_iter)

            if (cur_iter-1) % args.log_save_iter_freq == 0:
                msg = f"[iter - {cur_iter}]_[time - {time.time()-start_time}]_[D loss - {model.D_loss_val}]_[G loss - {model.G_loss_val}]_[r1 loss - {model.D_r1_loss_val}]"
                logger.write(msg)
            if cur_iter % args.img_save_iter_freq == 0:
                gene_img = model.inference()
                to_path = opj(args.img_save_dir, f"{cur_iter:08d}.png")
                img_save(gene_img, to_path, local_rank=args.local_rank)               
            if cur_iter % args.model_save_iter_freq == 0:
                to_path = opj(args.model_save_dir, f"{cur_iter:08d}.pth")
                model.save(to_path)
            cur_iter += 1


if __name__ == "__main__":
    args = build_args()
    logger = Logger(args.local_rank)
    logger.open(args.log_path)
    if args.use_DDP:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
    print_args(args, logger)
    main_worker(args, logger)
        
