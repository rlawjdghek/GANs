import sys

import torch
import numpy as np
import cv2

import torchvision

class Logger(object):
    def __init__(self, local_rank=0):
        self.terminal = sys.stdout
        self.file = None
        self.local_rank = local_rank
    def open(self, fp, mode=None):
        if mode is None: mode = 'w'
        self.file = open(fp, mode)
    def write(self, msg, is_terminal=1, is_file=1):
        if msg[-1] != "\n": msg = msg + "\n"
        if self.local_rank == 0:
            if '\r' in msg: is_file = 0
            if is_terminal == 1:
                self.terminal.write(msg)
                self.terminal.flush()
            if is_file == 1:
                self.file.write(msg)
                self.file.flush()
    def flush(self): 
        pass
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
  for k, v in vars(args).items():
      if logger is not None:
          logger.write('{:25s}: {}\n'.format(k, v))
      else:
          print('{:25s}: {}'.format(k, v))
def denorm(image):
    moco_input = np.zeros(image.shape)
    moco_input[:,:,0] = image[:,:,0] * 0.229 + 0.485
    moco_input[:,:,1] = image[:,:,1] * 0.224 + 0.456
    moco_input[:,:,2] = image[:,:,2] * 0.225 + 0.406
    return moco_input
def img_save(imgs, to_path, local_rank):
    imgs = torchvision.utils.make_grid(imgs, nrow=4, padding=0).cpu().detach().permute(1,2,0).numpy()
    imgs = denorm(imgs)
    imgs = np.uint8(imgs*255)
    if local_rank==0:
        cv2.imwrite(to_path, cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR))
    