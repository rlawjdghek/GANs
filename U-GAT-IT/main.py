import argparse
import os
from os.path import join as opj
import time
from datetime import datetime

import torch
import torch.distributed as dist
from torchvision.utils import make_grid

from dataset.dataloader import get_dataloader
from utils.utils import *
from models.UGATIT_model import UGATITModel

def build_args():
    parser = argparse.ArgumentParser()
    #### dataset ####
    parser.add_argument("--data_root_dir", type=str, default="/home/data")
    parser.add_argument("--data_name", type=str, default="selfie2anime")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--out_ch", type=int, default=3)
    parser.add_argument("--light", action="store_true")

    #### model ####
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--G_n_down", type=int, default=2)
    parser.add_argument("--G_n_blocks", type=int, default=3)  # default는 4
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--D_local_n_layers", type=int, default=5)
    parser.add_argument("--D_global_n_layers", type=int, default=7)

    #### train ####
    parser.add_argument("--n_iters", type=int, default=1000000)
    parser.add_argument("--G_lr", type=float, default=1e-4)
    parser.add_argument("--D_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_decay_iter", type=int, default=500000)
    parser.add_argument("--lambda_adv", type=float, default=1)
    parser.add_argument("--lambda_cycle", type=float, default=10)
    parser.add_argument("--lambda_id", type=float, default=10)
    parser.add_argument("--lambda_cam", type=float, default=1000)
    #### save & load ####
    parser.add_argument("--save_root_dir", type=str, default=f"/media/data1/jeonghokim/GANs/UGATIT/save/{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--train_print_iter_freq", type=int, default=10)
    parser.add_argument("--train_img_save_iter_freq", type=int, default=500)
    parser.add_argument("--n_train_img_save", type=int, default=8)
    parser.add_argument("--train_model_save_iter_freq", type=int, default=500)
    args = parser.parse_args()

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.logger_path = opj(args.save_root_dir, "log.txt")
    args.img_save_dir = opj(args.save_root_dir, "save_imgs")
    args.model_save_dir = opj(args.save_root_dir, "save_models")
    os.makedirs(args.img_save_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)

    return args

args = build_args()
logger = Logger(args.local_rank)
logger.open(args.logger_path)

torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend="nccl")

train_loader, test_loader = get_dataloader(args, logger)
model = UGATITModel(args, logger)

n_epochs = args.n_iters // len(train_loader)
cur_iter = 0
start_time = time.time()
for epoch in range(1, n_epochs+1):
    for img_A, img_B in train_loader:
        cur_iter += 1
        img_A = img_A.cuda()
        img_B = img_B.cuda()
        model.set_input(img_A, img_B)
        model.train(cur_iter)
        if cur_iter % args.train_print_iter_freq == 0:
            msg = f"[iter - {cur_iter}]_[time - {time.time() - start_time}]_[G_loss - {model.G_loss}]_[D_loss - {model.D_loss}]\n"
            logger.write(msg)

        #### save ####
        if args.local_rank == 0:
            #### image save ####
            if cur_iter % args.train_img_save_iter_freq == 0:
                real_imgs_A = []
                real_imgs_B = []
                gene_imgs_A = []
                gene_imgs_B = []
                G_heatmaps_A = []
                G_heatmaps_B = []
                D_global_heatmaps_A = []
                D_global_heatmaps_B = []
                D_local_heatmaps_A = []
                D_local_heatmaps_B = []
                f_resize = torchvision.transforms.Resize((args.img_size, args.img_size))
                model.to_eval()
                for i, (img_A, img_B) in enumerate(train_loader):
                    if i == args.n_train_img_save: break
                    img_A = img_A.cuda()
                    img_B = img_B.cuda()

                    gene_A, G_heatmap_A, gene_B, G_heatmap_B, D_global_heatmap_A, D_global_heatmap_B, D_local_heatmap_A, D_local_heatmap_B = model.inference(img_A, img_B)

                    real_imgs_A.append(img_A.cpu().detach())
                    real_imgs_B.append(img_B.cpu().detach())                
                    gene_imgs_A.append(gene_A.cpu().detach())
                    gene_imgs_B.append(gene_B.cpu().detach())
                    G_heatmaps_A.append(f_resize(G_heatmap_A.cpu().detach()))
                    G_heatmaps_B.append(f_resize(G_heatmap_B.cpu().detach()))
                    D_global_heatmaps_A.append(f_resize(D_global_heatmap_A.cpu().detach()))
                    D_global_heatmaps_B.append(f_resize(D_global_heatmap_B.cpu().detach()))
                    D_local_heatmaps_A.append(f_resize(D_local_heatmap_A.cpu().detach()))
                    D_local_heatmaps_B.append(f_resize(D_local_heatmap_B.cpu().detach()))
                model.to_train()
                real_imgs_A = make_grid(torch.vstack(real_imgs_A), nrow=1, padding=0)
                real_imgs_A = np.uint8(denorm(real_imgs_A.numpy().transpose(1,2,0))*255.0)
                real_imgs_B = make_grid(torch.vstack(real_imgs_B), nrow=1, padding=0)
                real_imgs_B = np.uint8(denorm(real_imgs_B.numpy().transpose(1,2,0))*255.0)
                gene_imgs_A = make_grid(torch.vstack(gene_imgs_A), nrow=1, padding=0)
                gene_imgs_A = np.uint8(denorm(gene_imgs_A.numpy().transpose(1,2,0))*255.0)
                gene_imgs_B = make_grid(torch.vstack(gene_imgs_B), nrow=1, padding=0)
                gene_imgs_B = np.uint8(denorm(gene_imgs_B.numpy().transpose(1,2,0))*255.0)
                G_heatmaps_A = make_grid(torch.vstack(G_heatmaps_A), nrow=1, padding=0).numpy().transpose(1,2,0)
                G_heatmaps_A = make_heatmap(G_heatmaps_A)
                G_heatmaps_B = make_grid(torch.vstack(G_heatmaps_B), nrow=1, padding=0).numpy().transpose(1,2,0)
                G_heatmaps_B = make_heatmap(G_heatmaps_B)
                D_global_heatmaps_A = make_grid(torch.vstack(D_global_heatmaps_A), nrow=1, padding=0).numpy().transpose(1,2,0)
                D_global_heatmaps_A = make_heatmap(D_global_heatmaps_A)
                D_global_heatmaps_B = make_grid(torch.vstack(D_global_heatmaps_B), nrow=1, padding=0).numpy().transpose(1,2,0)
                D_global_heatmaps_B = make_heatmap(D_global_heatmaps_B)
                D_local_heatmaps_A = make_grid(torch.vstack(D_local_heatmaps_A), nrow=1, padding=0).numpy().transpose(1,2,0)
                D_local_heatmaps_A = make_heatmap(D_local_heatmaps_A)
                D_local_heatmaps_B = make_grid(torch.vstack(D_local_heatmaps_B), nrow=1, padding=0).numpy().transpose(1,2,0)
                D_local_heatmaps_B = make_heatmap(D_local_heatmaps_B) 
                save_img_A = np.concatenate([real_imgs_B, gene_imgs_A, real_imgs_A, G_heatmaps_A, D_global_heatmaps_A, D_local_heatmaps_A], axis=1)
                save_img_B = np.concatenate([real_imgs_A, gene_imgs_B, real_imgs_B, G_heatmaps_B, D_global_heatmaps_B, D_local_heatmaps_B], axis=1)
                path_A = opj(args.img_save_dir, f"A2B_{cur_iter}.png")
                path_B = opj(args.img_save_dir, f"B2A_{cur_iter}.png")
                img_save(save_img_A, path_A)
                img_save(save_img_B, path_B)
            #### model save ####
            if cur_iter % args.train_model_save_iter_freq == 0:  
                to_path = opj(args.model_save_dir, f"[iter-{cur_iter}].pth")
                model.save(cur_iter=cur_iter, to_path=to_path)
            

