import argparse
import os
from os.path import join as opj
from datetime import datetime
import time

from utils.util import *
from datasets.dataloader import get_dataloader
from models.vid2vid import Vid2VidModel

def build_args(is_train=True):
    parser = argparse.ArgumentParser()

    #### dataset ####
    parser.add_argument("--data_root_dir", type=str, default="/home/data/CECTGAN")
    parser.add_argument("--data_name", type=str, default="CECT_all_vid")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--use_flip", type=bool, default=True)
    parser.add_argument("--use_crop", type=bool, default=False)
    parser.add_argument("--max_t_step", type=int, default=1, help="프레임을 얼마나 띄울건지. 1보다 크면 max_t_step과 1 사이의 값 중 랜덤하게 된다.")
    parser.add_argument("--input_ch", type=int, default=3)
    parser.add_argument("--output_ch", type=int, default=3)
    parser.add_argument("--n_workers", type=int, default=4)

    #### train ####
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--betas", default=(0.5,0.999))
    parser.add_argument("--n_frames_G", type=int ,default=3, help="# of frames to feed into G")
    parser.add_argument("--n_frames_D", type=int, default=3)
    parser.add_argument("--n_frames_total", type=int, default=30)
    parser.add_argument("--no_lsgan", action="store_true")
    parser.add_argument("--target_real_label", type=float, default=1.0)
    parser.add_argument("--target_gene_label", type=float, default=0.0)
    parser.add_argument("--n_scales_temporal", type=int, default=2, help="temporal D가 몇개인지. 지금까지 진행해온 프레임들로 중간에 몇개씩 스킵해서 학습을 몇번 할 건지.")
    parser.add_argument("--lambda_vgg", type=float, default=10.0, help="G를 위한 VGG loss")
    parser.add_argument("--lambda_feat", type=float, default=10.0, help="multiD를 통과한 pred_gene와 pred_real의 중간 feature loss")
    parser.add_argument("--lambda_warp", type=float, default=10.0, help="G의 warp loss")
    parser.add_argument("--lambda_flow", type=float, default=10.0, help='flow loss')

    #### model ####
    parser.add_argument("--G_name", type=str, default="official")
    parser.add_argument("--D_name" ,type=str, default="multi_scale")
    parser.add_argument("--ngf", type=int ,default=128)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--n_layers_D", type=int, default=3, help="NLayerD의 레이어 갯수")
    parser.add_argument("--n_D", type=int, default=2, help="multi scale을 위한 NLayersD갯수")
    parser.add_argument("--norm_type_D", type=str, default="bn")

    #### save ####
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--save_root_dir", type=str, default="/media/data1/jeonghokim/CECT/save/vid2vid")
    parser.add_argument("--save_name", type=str, default=f"{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--log_save_iter_freq", type=int, default=30)
    parser.add_argument("--img_save_iter_freq", type=int ,default=30)
    parser.add_argument("--model_save_iter_freq", type=int, default=1000)

    #### test ####
    parser.add_argument("--use_real_A", type=bool, default=True, help="이미지 생성할 때 이전의 B가 필요한데 처음에는 이게 없으니 A를 대신 사용. False이면 그냥 zero tensor")

    #### config ####
    parser.add_argument("--use_DDP", type=bool, default=False)

    args = parser.parse_args()
    if not is_train:
        args.use_DDP = False
        args.no_save = True
    
    if args.use_DDP: 
        args.n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else: 
        args.n_gpus = 1
        args.local_rank = 0
    args.save_dir = opj(args.save_root_dir, args.save_name)
    args.log_path = opj(args.save_dir, "log.txt")
    args.img_save_dir = opj(args.save_dir, "save_images")
    args.model_save_dir = opj(args.save_dir, "save_models")
    if not args.no_save:
        os.makedirs(args.img_save_dir, exist_ok=True)
        os.makedirs(args.model_save_dir, exist_ok=True)
    return args

def main_worker(args, logger):
    train_loader, valid_loader = get_dataloader(args)
    logger.write(f"1 epoch = {train_loader.dataset.sum_frames} iters")
    args.total_iters = int(args.n_epochs * train_loader.dataset.sum_frames)

    model = Vid2VidModel(args)

    cur_iter = 1
    start_time = time.time()
    print_args(args, logger)
    for epoch in range(args.n_epochs):
        loss_D_meter = AverageMeter()
        loss_G_meter = AverageMeter()
        loss_D_T_meter_lst = [AverageMeter() for _ in range(args.n_scales_temporal)]
        for v_idx, data in enumerate(train_loader):
            real_vA = data["A"].cuda(args.local_rank)  # 비디오 1개
            real_vB = data["B"].cuda(args.local_rank)  # 비디오 1개
            n_frames_total = data["n_frames_total"]
            is_first = True  # 맨처음 gene_Bs_prev, frames_all 없는거 명시
            for f_idx in range(n_frames_total-args.n_frames_G+1):
                real_As = real_vA[:, f_idx*args.input_ch:(f_idx+args.n_frames_G)*args.input_ch].reshape(args.batch_size, args.n_frames_G, args.input_ch, args.img_size, args.img_size)
                real_Bs = real_vB[:, f_idx*args.output_ch:(f_idx+args.n_frames_G)*args.output_ch].reshape(args.batch_size, args.n_frames_G, args.input_ch, args.img_size, args.img_size)

                model.set_input(real_As, real_Bs, is_first)
                model.train(cur_iter)

                BS = real_As.shape[0]
                loss_D_meter.update(model.loss_D.item(), BS)
                loss_G_meter.update(model.loss_G.item(), BS)
                for s in range(len(model.loss_D_T_lst)):
                    loss_D_T_meter_lst[s].update(model.loss_D_T_lst[s], BS)

                # TODO : 일단 생성 데이터 시각화
                if cur_iter % args.log_save_iter_freq == 0:
                    msg = f"[epoch - {epoch}/{args.n_epochs}]_[iter - {cur_iter}/{args.total_iters}]_[time - {time.time() - start_time:.2f}]_[loss G - {loss_G_meter.avg:.4f}]__[loss D - {loss_D_meter.avg:.4f}]"
                    for s in range(len(model.loss_D_T_lst)):
                        msg += f"_[loss D T_{s} - {loss_D_T_meter_lst[s].avg:.4f}]"
                    logger.write(msg)

                if cur_iter % args.img_save_iter_freq == 0:
                    _real_A_img = tensor2img(model.real_A_img)
                    _real_B_img = tensor2img(model.real_B_img)
                    _gene_B_img = tensor2img(model.gene_B_img)

                    _save_img = np.concatenate([_real_A_img, _real_B_img, _gene_B_img], axis=1)
                    to_path = opj(args.img_save_dir, f"[train]_[iter - {cur_iter}].png")
                    if args.local_rank == 0:
                        cv2.imwrite(to_path, _save_img)
                if cur_iter % args.model_save_iter_freq == 0:
                    to_path = opj(args.model_save_dir, f"[iter - {cur_iter}].pth")
                    model.save(to_path)

                is_first = False
                cur_iter += 1

if __name__ == "__main__":
    args = build_args()
    logger = Logger(args.local_rank)
    logger.open(args.log_path)

    main_worker(args, logger)
        

