# TransGAN Pytorch
Unofficial Pytorch implementation of TransGAN. CUDA환경 Distributed DataParallel에서만 구현하였다. 데이터는 CelebA_HQ 사용.

CUDA=11.3 <br/>
pytorch 1.11.1+cu113

### Training
```bash
torchrun --nproc_per_node <number of GPUs> train.py
```
