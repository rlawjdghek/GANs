import sys
import argparse
import json

import numpy as np

class Logger(object):
    def __init__(self, local_rank=0):
        self.terminal = sys.stdout
        self.file = None
        self.local_rank = local_rank
    def open(self, fp, mode=None):
        if mode is None: mode = 'w'
        if self.local_rank == 0: self.file = open(fp, mode)
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
def get_lr(optimizer):
    for g in optimizer.param_groups:
        return g["lr"]
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
def save_args(args, to_path):
    with open(to_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
def load_args(from_path):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(from_path, "r") as f:
        args.__dict__ = json.load(f)
    return args   
def tensor2img(x):
    x = x[0].permute(1,2,0).detach().cpu().numpy()
    x = (x+1)/2
    x = np.uint8(x*255.0)
    return x
  