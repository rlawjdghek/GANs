import argparse
import os
from os.path import join as opj

import torch

from models.networks import Generator
from utils.utils import img_save

def build_args():
    parser = argparse.ArgumentParser()   
    #### model ####
    parser.add_argument("--size", type=int, default=1024, help="image size")
    parser.add_argument("--style_dim", type=int, default=512, help="dimension for style")
    parser.add_argument("--n_mlp", type=int, default=8, help="# layers of mapping network")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="multiplier for channel dimension")

    #### train & test ####
    parser.add_argument("--truncation", type=int, default=0.1, help="truncation ratio")
    parser.add_argument("--truncation_mean", type=int, default=4096, help="# vectors to calculate mean for the truncation")
    #### save & Load ####
    parser.add_argument("--save_img_dir", default=f"/media/data1/jeonghokim/GANs/StyleGAN2/generate/20220206_27000_truncation0.1")
    parser.add_argument("--cp_load_path", type=str, default="/media/data1/jeonghokim/GANs/StyleGAN2/save/20220206_train/save_models/27000.pth")
    parser.add_argument("--n_samples", type=int, default=50, help="# samples to generate")
    
    args = parser.parse_args()
    os.makedirs(args.save_img_dir, exist_ok=True)
    return args
def generate(args, model, mean_style):
    with torch.no_grad():
        for iter in range(args.n_samples):
            sample_z = torch.randn(1, args.style_dim).cuda()
            gene_img, _ = model([sample_z], truncation=args.truncation, truncation_style=mean_style)
            to_path = opj(args.save_img_dir, f"{iter}.png")
            img_save(gene_img, to_path)

if __name__ == "__main__":
    args = build_args()
    G_ema = Generator(
        size=args.size,
        style_dim=args.style_dim,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier
    ).cuda()
    cp = torch.load(args.cp_load_path)
    G_ema.load_state_dict(cp["G_ema"])
    G_ema.eval()
    print(f"model is successfully loaded from {args.cp_load_path}")
    if args.truncation < 1:
        with torch.no_grad():
            mean_style = G_ema.mean_style(args.truncation_mean)
    else:
        mean_style = None
    generate(args, G_ema, mean_style)
    
    
    
    


