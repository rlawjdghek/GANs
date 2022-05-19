# StyleGAN2 pytorch
Unofficial Pytorch implementation of StyleGAN2. CUDA환경 Distributed DataParallel에서만 구현하였다.
CUDA=11.2
pytorch 1.7.1+cu110

### Training
```bash
python -m torch.distributed.launch --nproc_per_node <# GPUS> --master_port <master port> train.py
```


### Generation
```bash
python generate.py --save_img_dir <image save directory> --cp_load_path <checkpoint of G_ema> --n_samples <number of samples to generate> --truncation <truncation_ratio> --truncation_mean <# vectors for truncation mean vector>
```

truncation_mean 변수는 truncation을 하기위해 몇 개의 벡터를 사용할 것이냐를 나타냄. 많을수록 가우시안의 평균인 0에 가까워진다.
truncation 변수는 truncation_mean을 얼마나 사용하지 않을 것인지에 대한 숫자. 1이면 truncation을 아예 하지 않는 것이고 0이면 똑같은 이미지만 나온다.


