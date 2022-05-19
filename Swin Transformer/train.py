import argparse
import os
from os.path import join as opj
import time
from datetime import datetime

import torch.distributed as dist

from datasets.dataloader import get_dataloader
from models.swintransformer import SwinTransformer
from utils.utils import *
def build_args():
    parser = argparse.ArgumentParser()
    #### dataset ####
    parser.add_argument("--data_root_dir", type=str, default="/home/data")
    parser.add_argument("--data_type", type=str, default="Imagenet")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--n_workers", type=int, default=4)

    #### model ####
    parser.add_argument("--layer_init_type", type=str, default="xavier_uniform")

    #### train ####
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=str, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_lr", type=float, default=5e-7)
    parser.add_argument("--min_lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--betas", default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--warmup_epoch", type=int, default=20)
    
    #### save & load ####
    parser.add_argument("--save_root_dir", type=str, default=f"/media/data1/jeonghokim/swintransformer/save/{datetime.now().strftime('%Y%m%d')}")
    #### config ####
    parser.add_argument("--use_DDP", action="store_true", default=False)
    args = parser.parse_args()
    if args.use_DDP: args.local_rank = int(os.environ["LOCAL_RANK"])
    else: args.local_rank = 0
    args.logger_path = opj(args.save_root_dir, "log.txt")
    args.model_save_dir = opj(args.save_root_dir, "save_models")
    os.makedirs(args.model_save_dir, exist_ok=True)
    return args

def main_worker(args, logger):
    train_loader, valid_loader, train_sampler = get_dataloader(args)
    print(len(train_loader), len(valid_loader))

    model = SwinTransformer(args, logger, 1000)

    train_top1_acc = AverageMeter()
    train_top5_acc = AverageMeter()
    train_loss = AverageMeter()
    valid_top1_acc = AverageMeter()
    valid_top5_acc = AverageMeter()
    valid_loss = AverageMeter()
    start_time = time.time()
    for epoch in range(1, args.n_epochs+1):
        if args.use_DDP: train_sampler.set_epoch(epoch)
        for img, label in train_loader:
            BS = img.shape[0]
            img = img.cuda()
            label = label.cuda()
            model.set_input(img, label)
            output, loss = model.train()
            top1_acc, top5_acc = Accuracy(output, label, topk=(1,5))
            train_top1_acc.update(top1_acc.item(), BS)
            train_top5_acc.update(top5_acc.item(), BS)
            train_loss.update(loss.item(), BS)
        for img, label in valid_loader:
            BS = img.shape[0]
            img = img.cuda()
            label = label.cuda()
            model.set_input(img, label)
            output, loss = model.train()
            top1_acc, top5_acc = Accuracy(output, label, topk=(1,5))
            valid_top1_acc.update(top1_acc.item(), BS)
            valid_top5_acc.update(top5_acc.item(), BS)
            valid_loss.update(loss.item(), BS)
        msg = f"[Epoch - {epoch}]_[Time - {time.time() - start_time}]_[train loss - {train_loss.avg:.4f}]_[train top1 acc - {train_top1_acc.avg:.4f}_[train top5 acc - {train_top5_acc.avg:.4f}]_[valid loss - {valid_loss.avg:.4f}]_[valid top1 acc - {valid_top1_acc.avg:.4f}]_[valid top5 acc - {valid_top5_acc.avg:.4f}]"
        logger.write(msg)


if __name__=="__main__":
    args = build_args()
    logger = Logger(local_rank=args.local_rank)
    logger.open(args.logger_path)
    print_args(args, logger)
    if args.use_DDP:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_grou