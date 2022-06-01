import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
asd= torch.nn.Linear(10,100).cuda()