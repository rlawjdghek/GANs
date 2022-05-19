import argparse
import os
from os.path import join as opj

import wandb

from util.utils import *
from dataset.dataloader import get_dataloader
from model.cyclegan import CycleGANModel

def build_args():
    parser = argparse.ArgumentParser()
    #### dataset ####
    parser.add_argument("--data_root_dir", type=str, default="/home/data/")
    parser.add_argument("--data_name", type=str, default="CECT_all")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--input_nc", type=int, default=3)
    parser.add_argument("--output_nc", type=int, default=3)
    parser.add_argument("--resize", type=bool, default=True)
    parser.add_argument("--crop", type=bool, default=True)
    parser.add_argument("--size", type=int, default=286)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--flip", type=bool, default=True)
    parser.add_argument("--input_ch", type=int, default=3)
    parser.add_argument("--output_ch", type=int, default=3)
    parser.add_argument("--pool_size", type=int, default=50)
    parser.add_argument("--AtoB_dir", type=int, default="ncct")
    parser.add_argument("--BtoA_dir", type=str, default="cect")

    #### train & test ####
    parser.add_argument("--batch_size", type=int, default=1)  ####
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_scheduler", type=str, default="linear",
                        choices=["linear", "step", "plateau", "cosine"])
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--linearlr_epochs", type=int, default=50, help="linear decay ratio for linear lr scheduler")
    parser.add_argument("--steplr_step", type=int, default=50)
    parser.add_argument("--gan_loss_name", type=str, default="lsgan",
                        choices=["lsgan", "wgangp", "vanilla"])
    parser.add_argument("--target_real_label", type=float, default=1.0)
    parser.add_argument("--target_gene_label", type=float, default=0.0)
    parser.add_argument("--G_lr", type=float, default=2e-4)
    parser.add_argument("--D_lr", type=float, default=2e-4)
    parser.add_argument("--G_betas", type=tuple, default=(0.5, 0.999))
    parser.add_argument("--D_betas", type=tuple, default=(0.5, 0.999))

    parser.add_argument("--lambda_ID", type=float, default=0.5)
    parser.add_argument("--lambda_A", type=float, default=10.0)
    parser.add_argument("--lambda_B", type=float, default=10.0)

    #### model ####
    parser.add_argument("--G_AB_name", type=str, default="unet_256",
                        choices=["resnet_6blks", "resnet_9blks", "unet_128", "unet_256", "unet_512"])
    parser.add_argument("--G_BA_name", type=str, default="unet_256",
                        choices=["resnet_6blks", "resnet_9blks", "unet_128", "unet_256", "unet_512"])
    parser.add_argument("--D_A_name", type=str, default="basic",
                        choices=["basic", "n_layers", "pixel"])
    parser.add_argument("--D_B_name", type=str, default="basic",
                        choices=["basic", "n_layers", "pixel"])
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int ,default=64)
    parser.add_argument("--G_norm_type", type=str, default="instance",
                        choices=["batch", "instance", "none"])
    parser.add_argument("--D_norm_type", type=str, default="instance",
                        choices=["batch", "instance", "none"])
    parser.add_argument("--G_init_type", type=str, default="normal",
                        choices=["normal", "xavier", "kaiming", "orthogonal"])
    parser.add_argument("--D_init_type", type=str, default="normal",
                        choices=["normal", "xavier", "kaiming", "orthogonal"])
    parser.add_argument("--G_init_gain", type=float, default=0.02)
    parser.add_argument("--D_init_gain", type=float, default=0.02)
    parser.add_argument("--D_n_layers", type=int, default=3)
    parser.add_argument("--G_use_dropout", type=bool, default=False)

    #### save & load ####
    parser.add_argument("--save_root_dir", type=str, default="/home/jeonghokim/")
    parser.add_argument("--save_name", type=str, default="bigdata_test")  ####
    parser.add_argument("--img_save_iter_freq", type=int, default=1000)
    parser.add_argument("--model_save_iter_freq", type=int, default=1000)
    parser.add_argument("--n_save_images", type=int, default=8)
    parser.add_argument("--msssim_epoch_freq", type=int, default=9999999, help="train, validation calculating msssim")
    parser.add_argument("--valid_epoch_freq", type=int, default=20)
    
    #### config ####
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--DP", action="store_true")
    parser.add_argument("--DDP", action="store_true", default=True)
    parser.add_argument("--dist_backend", default="nccl")
    parser.add_argument("--use_wandb", action="store_true")
    # parser.add_argument("--wandb_notes", type=str, default="_exp")

    args = parser.parse_args()
    args.save_dir = opj(args.save_root_dir, args.save_name)
    args.img_save_dir = opj(args.save_dir, "save_images")
    args.model_save_dir = opj(args.save_dir, "save_models")
    args.logger_path = opj(args.save_dir, "log.txt")
    args.save_name = f"[G_AB-{args.G_AB_name}]_[G_BA-{args.G_BA_name}]_[D_A-{args.D_A_name}]_" \
                     f"[D_B-{args.D_B_name}]"
    args.wandb_name = args.save_name
    args.wandb_notes = args.save_name
    os.makedirs(args.img_save_dir + "/NtoC", exist_ok=True)
    os.makedirs(args.img_save_dir + "/CtoN", exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    return args
def train(args, logger):
    train_loader, valid_loader = get_dataloader(args, logger)
    args.total_iter = args.n_epochs * len(train_loader)
    CycleGAN = CycleGANModel(args, logger)
    iteration = 1
    for epoch in range(args.start_epoch, args.n_epochs+1):
        CycleGAN.reset_meters()
        #### training ####
        for data in train_loader:
            ncct_img = data["ncct_img"].cuda(args.local_rank)
            cect_img = data["cect_img"].cuda(args.local_rank)
            CycleGAN.set_input(ncct_img, cect_img)
            CycleGAN.train(iteration)
            iteration += 1
        CycleGAN.scheduler_G.step()
        CycleGAN.scheduler_D.step()
        #### Validation 이미지저장 ####
        if epoch % args.valid_epoch_freq == 0:
            CycleGAN.to_eval()
            for _iter, data in enumerate(valid_loader):
                ncct_img = data["ncct_img"].cuda(args.local_rank)
                cect_img = data["cect_img"].cuda(args.local_rank)
                CycleGAN.set_input(ncct_img, cect_img)
                CycleGAN.validation(epoch, _iter)
            CycleGAN.to_train()
        if args.use_wandb and args.local_rank == 0:
            wandb_msg = {"loss G": CycleGAN.train_loss_G.avg,
                         "loss D": CycleGAN.train_loss_D.avg}
            wandb.log(wandb_msg)
        logger.write(f"[Epoch-{epoch}] loss G: {CycleGAN.train_loss_G.avg}, loss D: {CycleGAN.train_loss_D.avg}\n")
        #### MSSSIM, PSNR 측정 ####
        if epoch % args.msssim_epoch_freq == 0:
            CycleGAN.to_eval()
            for data in train_loader:
                ncct_img = data["ncct_img"].cuda(args.local_rank)
                cect_img = data["cect_img"].cuda(args.local_rank)
                CycleGAN.set_input(ncct_img, cect_img)
                CycleGAN.calc_msssim_psnr(is_train=True)

            for data in valid_loader:
                ncct_img = data["ncct_img"].cuda(args.local_rank)
                cect_img = data["cect_img"].cuda(args.local_rank)
                CycleGAN.set_input(ncct_img, cect_img)
                CycleGAN.calc_msssim_psnr(is_train=False)
            if args.use_wandb and args.local_rank==0:
                wandb_msg = {"train MSSSIM NCCT": CycleGAN.train_msssim_A.avg,
                             "train MSSSIM CECT": CycleGAN.train_msssim_B.avg,
                             "train PSNR NCCT": CycleGAN.train_psnr_A.avg,
                             "train PSNR CECT": CycleGAN.train_psnr_B.avg,
                             "valid MSSSIM NCCT": CycleGAN.valid_msssim_A.avg,
                             "valid MSSSIM CECT": CycleGAN.valid_msssim_B.avg,
                             "valid PSNR NCCT": CycleGAN.valid_psnr_A.avg,
                             "valid PSNR CECT": CycleGAN.valid_psnr_B.avg}
                wandb.log(wandb_msg)
            logger.write(f"[Epoch-{epoch}]_{wandb_msg}\n")
            CycleGAN.to_train()
if __name__ == "__main__":
    args = build_args()
    logger = Logger(args.local_rank)
    logger.open(args.logger_path)
    print_args(args, logger)
    if args.use_wandb and args.local_rank==0:
        wandb.init(project="CECT CycleGAN", name=args.wandb_name, notes=args.wandb_notes)
        wandb.config.update(args)
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend=args.dist_backend, world_size=args.world_size)

    train(args, logger)
