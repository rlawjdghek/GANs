import os
from os.path import join as opj
import argparse
import time
from datetime import datetime

import wandb
import torch
import torch.distributed as dist

from datasets.dataloader import get_dataloader
from utils.utils import *
from models.stylegan2_model import StyleGAN2Model

def build_args():
    parser = argparse.ArgumentParser()
    #### dataset ####
    parser.add_argument("--data_root_dir", type=str, default="/home/data/FFHQ_1024")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--n_iters", type=int, default=800000)
    #### model ####
    parser.add_argument("--size", type=int, default=1024, help="image size")
    parser.add_argument("--style_dim", type=int, default=512, help="dimension for style")
    parser.add_argument("--n_mlp", type=int, default=8, help="# layers of mapping network")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="multiplier for channel dimension")
    #### train & test ####
    parser.add_argument("--reg_every_G", type=int, default=4, help="interval of the applying path length  regularization")
    parser.add_argument("--reg_every_D", type=int, default=16, help="interval of the applying r1 regularization")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--augment_p", type=float, default=0, help="probability of applying augmentation. 0 use adaptive augmentation")
    parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")
    parser.add_argument("--ada_target", type=float, default=0.6, help="target augmentation probability for adaptive augmentation")
    parser.add_argument("--mixing", type=float, default=0.9, help="probability of style mixing")
    parser.add_argument("--n_samples", type=int, default=16, help="# samples generated during training")
    parser.add_argument("--path_batch_shrink",type=int, default=2, help="batch size reducing factor for the path length regularization (reduce memory comsumption)")
    parser.add_argument("--w_path_regularize",type=float, default=2, help="weight of the path length regularization")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    #### save & load ####
    parser.add_argument("--cp_path", default=None)
    parser.add_argument("--save_root_dir", type=str, default=f"/media/data1/jeonghokim/GANs/StyleGAN2/save/{datetime.now().strftime('%Y%m%d')}_train")
    parser.add_argument("--log_save_iter_freq", type=int, default=100)
    parser.add_argument("--img_save_iter_freq", type=int, default=1000)
    parser.add_argument("--model_save_iter_freq", type=int, default=1000)
    #### config ####
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--wandb_name", type=str, default=f"{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--wandb_notes", type=str, default="test")
    parser.add_argument("--DDP_backend", type=str, default="nccl")
    args = parser.parse_args()
    args.n_workers //= args.world_size
    args.batch_size //= args.world_size

    args.logger_path = opj(args.save_root_dir, "logger.txt")
    args.save_img_dir = opj(args.save_root_dir, "save_images")
    args.save_model_dir = opj(args.save_root_dir, "save_models")
    os.makedirs(args.save_img_dir, exist_ok=True)
    os.makedirs(args.save_model_dir, exist_ok=True)
    return args

def train(args, logger):
    train_loader = get_dataloader(args, logger)
    model = StyleGAN2Model(args, logger)
    start_iter = model.start_iter
    start_time = time.time()
    for iter in range(start_iter, args.n_iters):
        real_img = next(train_loader).to(args.local_rank)
        model.set_input(real_img)
        model.train(iter)
        if args.local_rank == 0:
            if iter % args.log_save_iter_freq == 0:
                logger.write(
                f"[Iteration-{iter}/{args.n_iters}]_[Running Time-{time.time() - start_time:.2f}s]_\
                [Loss D-{model.loss_val_D}_[Loss G-{model.loss_val_G}]_\
                [Loss Path-{model.path_loss_val}]_[Mean Loss Path-{model.mean_path_length_avg}]_\
                [Augment-{model.ada_aug_p}]"
                )   
                if args.use_wandb:
                    wandb.log(
                        {
                            "Loss G": model.loss_val_G,
                            "Loss D": model.loss_val_D,
                            "Real Loss D": model.real_loss_val_D,
                            "Gene Loss D": model.gene_loss_val_D,
                            "Augment": model.ada_aug_p,
                            "Rt": model.r_t_stat,
                            "R1": model.r1_loss_val,
                            "Path Length Regularization": model.path_loss_val,
                            "Mean Path Length": model.mean_path_length,
                            "Path Length": model.path_length_val
                        }
                    )
            if iter % args.img_save_iter_freq == 0:
                to_path = opj(args.save_img_dir, f"{iter}.png")
                test_sample = model.inference()
                img_save(test_sample, to_path)
            if iter % args.model_save_iter_freq == 0:
                to_path = opj(args.save_model_dir, f"{iter}.pth")
                model.save(iter, to_path)
        dist.barrier()

if __name__=="__main__":
    args = build_args()
    logger = Logger(args.local_rank)
    logger.open(args.logger_path)
    if args.use_wandb and args.local_rank == 0:
        wandb.init(project="StyleGAN2", name=args.wandb_name, notes=args.wandb_notes)
        wandb.config.update(args)
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.DDP_backend, world_size=args.world_size)
    print_args(args, logger)
    train(args, logger)
    

        