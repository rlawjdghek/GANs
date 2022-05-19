import argparse
import os
from os.path import join as opj
from datetime import datetime
import time

import wandb
import torch
import torch.distributed as dist

from utils.utils import *
from datasets.dataloader import get_dataloader
from models.transgan_model import TransGANModel

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
    #### dataset ###
    parser.add_argument("--data_root_dir", type=str, default="/home/data/")
    parser.add_argument("--data_type", type=str, default="celebA")
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--diff_aug", type=str, default="filter,translation,erase_ratio,color,hue", help="augmentations for D")

    #### train & test ####
    parser.add_argument("--batch_size", type=int, default=2, help="G 학습할때에는 2배로 들어감.") 
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--init_size", type=int, default=8, help="initial noise size")
    parser.add_argument("--n_iters", type=int, default=500000)
    parser.add_argument("--D_iter_per_G", type=int, default=4, help="train iter for D per one iter of G")
    parser.add_argument("--G_optim", type=str, default="adam")
    parser.add_argument("--D_optim", type=str, default="adam")
    parser.add_argument("--G_lr", type=float, default=1e-4)
    parser.add_argument("--D_lr", type=float, default=1e-4)
    parser.add_argument("--G_beta1", type=float, default=0)
    parser.add_argument("--G_beta2", type=float, default=0.99)
    parser.add_argument("--D_beta1", type=float, default=0)
    parser.add_argument("--D_beta2", type=float, default=0.99)
    parser.add_argument("--G_weight_decay", type=float, default=1e-3)
    parser.add_argument("--D_weight_decay", type=float, default=1e-3)
    parser.add_argument('--accumulated_times', type=int, default=4, help='gradient accumulation')
    parser.add_argument('--G_accumulated_times', type=int, default=4, help='gradient accumulation')
    parser.add_argument("--loss_type", type=str, default="wgangp-eps", choices=["hinge", "lsgan", "wgangp", "wgangp-mode", "wgangp-eps", "standard"])
    parser.add_argument("--wgangp_phi", type=float, default=1)
    parser.add_argument("--ema", type=float, default=0.995, help="exponential moving average")
    parser.add_argument("--ema_k_img", type=float, default=500, help="단위가 1000임. 나중에 1000곱해짐")
    parser.add_argument("--ema_warmup", type=float, default=0)

    #### model ####
    parser.add_argument("--layer_init_type", type=str, default="xavier_uniform", choices=["normal", "orthogonal", "xavier_uniform"])
    parser.add_argument("--bottom_width", type=int, default=8)
    parser.add_argument("--G_embedding_dim", type=int, default=1024, help="base embedding dimension of G")
    parser.add_argument("--D_embedding_dim", type=int, default=384, help="base embedding dimension of D")
    parser.add_argument("--G_n_heads", type=int, default=4)
    parser.add_argument("--D_n_heads", type=int, default=4)
    parser.add_argument("--G_window_size", type=int, default=16)
    parser.add_argument("--D_window_size", type=int, default=4)
    parser.add_argument("--G_norm_type", type=str, default="pn", choices=["ln", "bn", "in", "pn"])
    parser.add_argument("--D_norm_type", type=str, default="ln")
    parser.add_argument("--G_act_type", type=str, default="gelu", choices=["gelu", "lrelu"])
    parser.add_argument("--D_act_type", type=str, default="gelu")
    parser.add_argument("--G_depths", default="5,4,4,4,4,4")
    parser.add_argument("--D_depths", type=str, default=3)
    parser.add_argument("--G_mlp_ratio", type=str, default=4, help="G mlp ratio")
    parser.add_argument("--D_mlp_ratio", type=str, default=4, help="D mlp ratio")
    parser.add_argument("--D_attn_drop", type=float, default=0.0)
    parser.add_argument("--D_proj_drop", type=float, default=0.0)
    parser.add_argument("--D_mlp_drop", type=float, default=0.0)
    parser.add_argument("--D_drop_path_rate", type=float, default=0.0)
    parser.add_argument("--D_in_ch", type=int, default=3)
    parser.add_argument("--latent_norm", action="store_true", help="use normalization of initial noise z")
    parser.add_argument("--G_out_ch", type=int, default=3)
    parser.add_argument("--n_cls", type=int, default=1)
    parser.add_argument("--D_patch_size", type=int, default=2)
    #### save & load ####
    parser.add_argument("--save_root_dir", type=str, default=f"/data/jeonghokim/GANs/TransGAN/save/{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--log_save_iter_freq", type=int, default=100)
    parser.add_argument("--img_save_iter_freq", type=int, default=500)
    parser.add_argument("--model_save_iter_freq", type=int, default=10000)  # 모델 2.5GB
    
    #### config #### 
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--wandb_name" ,type=bool, default=f"{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--wandb_notes", type=str, default="test")
    parser.add_argument("--DDP_backend", type=str, default="nccl")
    args = parser.parse_args()
    args.G_depths = list(map(int, args.G_depths.split(",")))
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.logger_path = opj(args.save_root_dir, "log.txt")
    args.img_save_dir = opj(args.save_root_dir, "save_images")
    args.model_save_dir = opj(args.save_root_dir, "save_models")
    os.makedirs(args.img_save_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)    
    return args

def main_worker(args, logger):
    train_loader, train_sampler = get_dataloader(args)
    model = TransGANModel(args, logger)
    n_epochs = args.n_iters // len(train_loader)
    import time
    start_time = time.time()
    cur_iter = 1
    for epoch in range(n_epochs+1):
        train_sampler.set_epoch(epoch)
        for img in train_loader:
            img = img.cuda(args.local_rank, non_blocking=True)
            z = torch.randn((img.shape[0], args.latent_dim)).cuda(args.local_rank, non_blocking=True)
            model.set_input(img, z)
            model.train(cur_iter)
            #### save ####
            if True:
                if (cur_iter-1) % args.log_save_iter_freq == 0:
                    D_loss = model.D_loss.item()
                    G_loss = model.G_loss.item()
                    logger.write(f"[iteration - {cur_iter}/{args.n_iters}]_[time - {time.time()-start_time:.4f}]_[D loss - {D_loss}]_[G loss - {G_loss}]\n")
                if cur_iter % args.img_save_iter_freq == 0:
                    to_path = opj(args.img_save_dir, f"{cur_iter}.png")
                    sample_img = model.inference()
                    img_save(sample_img, to_path, args.local_rank)
                if cur_iter % args.model_save_iter_freq == 0: 
                    to_path = opj(args.model_save_dir, f"{cur_iter}.pth")
                    model.save(to_path)
            cur_iter += 1
        dist.barrier()
if __name__=="__main__":
    args = build_args() 
    logger = Logger(local_rank=args.local_rank)
    if args.use_wandb and args.local_rank == 0:
        wandb.init(project="TransGAN", name=args.wandb_name, notes=args.wandb_notes)
        wandb.config.update(args)
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.DDP_backend)
    logger.open(args.logger_path)
    print_args(args, logger)
    main_worker(args, logger)
