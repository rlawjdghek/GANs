import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from os.path import join as opj
import torch
import numpy as np
import cv2

from train import build_args
from models.base_module import define_module_G
from datasets.dataloader import get_test_dataloader
from utils.util import tensor2img, tensor2flow

if __name__ == "__main__":
    args = build_args(is_train=False)

    #### 수정 ####
    args.use_real_A = True
    torv = "test"
    date = "20220514"
    test_data_name = "20220210_test"
    load_path = f"/media/data1/jeonghokim/CECT/save/vid2vid/{date}/save_models/[iter - 342].pth"
    root_save_dir = f"/media/data1/jeonghokim/CECT/save/vid2vid/{date}/inference_images/{test_data_name if torv=='test' else 'valid'}"
    os.makedirs(root_save_dir, exist_ok=True)
    ##############

    model = define_module_G(args).cuda()
    model.load(load_path)
    model.to_eval()

    valid_loader, test_loader = get_test_dataloader(args, test_data_name)
    
    if torv == "test": loader = test_loader
    elif torv == "valid": loader = valid_loader
    for v_idx, data in enumerate(loader):
        real_vA = data['A'].cuda()
        real_vB = data['B'].cuda()
        print(real_vA.shape)
        print(real_vB.shape)
        n_frames_total = real_vA.shape[1]//args.input_ch
        if args.use_real_A:  # 처음에는 이전 예측한 이미지가 없음.
            gene_Bs_prev = real_vA[:, :(args.n_frames_G-1)*args.input_ch].reshape(1, args.n_frames_G-1, args.input_ch, args.img_size, args.img_size)
        else:
            gene_Bs_prev = torch.zeros_like(real_vA[:, :(args.n_frames_G-1)*args.input_ch]).reshape(1, args.n_frames_G-1, args.input_ch, args.img_size, args.img_size)
        for f_idx in range(n_frames_total-args.n_frames_G+1):
            real_As = real_vA[:, f_idx*args.input_ch:(f_idx+args.n_frames_G)*args.input_ch].reshape(1, args.n_frames_G, args.input_ch, args.img_size, args.img_size)
            real_Bs = real_vB[:, f_idx*args.input_ch:(f_idx+args.n_frames_G)*args.input_ch].reshape(1, args.n_frames_G, args.input_ch, args.img_size, args.img_size)

            gene_B, gene_B_raw, gene_flow, gene_weight = model.inference(real_As, gene_Bs_prev)
            gene_Bs_prev = torch.cat([gene_Bs_prev[:, 1:], gene_B.unsqueeze(1)], dim=1)
            real_A = real_As[:, -1]
            real_B = real_Bs[:, -1]
            real_A_img = tensor2img(real_A)
            real_B_img = tensor2img(real_B)
            gene_B_img = tensor2img(gene_B)
            gene_B_raw_img = tensor2img(gene_B_raw)
            gene_flow_img = tensor2flow(gene_flow)
            gene_weight_img = tensor2img(gene_weight)

            save_img = np.concatenate([real_A_img, real_B_img, gene_B_img, gene_B_raw_img, gene_flow_img, gene_weight_img], axis=1)
            save_dir = opj(root_save_dir, f"{v_idx:06d}")
            os.makedirs(save_dir, exist_ok=True)
            if f_idx == 0:  # 첫장으로 이전 프레임들 복사하기
                for i in range(args.n_frames_G-1):
                    to_path = opj(save_dir, f"{f_idx:06d}.png")
                    cv2.imwrite(to_path, save_img)
            to_path = opj(save_dir, f"{f_idx+args.n_frames_G-1:06d}.png")
            cv2.imwrite(to_path, save_img)