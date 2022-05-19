# U-GAT-IT pytorch
Unofficial Pytorch implementation of UGATIT. CUDA환경 Distributed DataParallel에서만 구현하였다. 데이터는 selfie2anime를 활용 [링크](https://www.kaggle.com/arnaud58/selfie2anime). 

3090에 모델이 다 안올라가서 generator의 resblock을 하나 줄였다.
CUDA=11.2 <br/>
pytorch 1.10.0+cu110

### Training
```bash
torchrun --nproc_per_node <number of GPUs> main.py
```
