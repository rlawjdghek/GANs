# StarGANv2 Pytorch
Unofficial Pytorch implementation of StarGANv2 for AFHQ

Implement FID and LPIPS scores for latent and reference-guided generated images.

CUDA=11.3 <br/>
pytorch 1.11.1+cu113

### Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py --use_DDP True
```
