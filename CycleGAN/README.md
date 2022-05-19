# CycleGAN for ncct -> cect 
강북병원 프로젝트 베이스라인으로 사용된 CycleGAN. Official code 기반으로 재구성하였다.
기본 모델설정은 Generator : UNet_256, Discriminator : 단순 CNN



### Training & Validation
```bash
CUDA_VISIBLE_DEVICES=<your gpu ids> python -m torch.distributed.launch --nproc_per_node <\# gpus> main.py --data_root_dir <your data directory> --data_name <data name> --AtoB_dir <source data folder name> --BtoA_dir <target data folder name> --valid_epoch_freq <how many epochs to valid>
```