import sys
import torch
import cv2
import numpy as np
import os
import json
import argparse

class Logger(object):
    def __init__(self, local_rank=0):
        self.terminal = sys.stdout
        self.file = None
        self.local_rank = local_rank
    def open(self, fp, mode=None):
        self.fp = fp
        if mode is None: mode = 'w'
        if self.local_rank == 0:
            self.file = open(fp, mode)
    def write(self, msg, is_terminal=1, is_file=1):
        if msg[-1] != "\n": msg = msg + "\n"
        if self.local_rank == 0:
            if '\r' in msg: is_file = 0
            if is_terminal == 1:
                self.terminal.write(msg)
                self.terminal.flush()
            if is_file == 1:
                if os.path.exists(self.fp):
                    self.file.write(msg)
                    self.file.flush()
    def flush(self): 
        pass
def save_args(args, to_path):
    with open(to_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
def load_args(from_path):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(from_path, "r") as f:
        args.__dict__ = json.load(f)
    return args  
class AverageMeter (object):
    def __init__(self):
        self.reset ()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def print_args(args, logger=None):
    if logger is not None:
        logger.write("#### configurations ####")
    for k, v in vars(args).items():
        if logger is not None:
            logger.write('{}: {}\n'.format(k, v))
        else:
            print('{}: {}'.format(k, v))
    if logger is not None:
        logger.write("########################")
def img_read(path, size):
    img = cv2.imread(path).astype(np.float32) / 255.0
    img = cv2.resize(img, (size, size))
    return img
def psnr(output, target):
    assert output.shape == target.shape, "input image shape must have same shape"
    output = torch.clip(output, 0, 1)
    target = torch.clip(target, 0, 1)
    mse = torch.mean((output - target) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
def tensor2img(x):
    if x.shape[1] != 3:
        x = torch.cat([x,x,x], dim=1)
    x = x[0].permute(1,2,0).cpu().detach().numpy()
    x = (x+1)/2
    x = np.clip(x, 0, 1)
    x = np.uint8(x*255.0)
    return x
def tensor2flow(x):
    x = x[0].permute(1,2,0).cpu().detach().float().numpy()
    hsv = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
    hsv[:,:,0] = 255
    hsv[:,:,1] = 255
    mag, ang = cv2.cartToPolar(x[:,:,0], x[:,:,1])
    hsv[:,:,0] = ang * 180 / np.pi / 2
    hsv[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


    
