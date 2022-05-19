import sys

import numpy as np
import cv2
import torchvision

class Logger(object):
    def __init__(self, local_rank=0):
        self.terminal = sys.stdout
        self.file = None
        self.local_rank=local_rank
    def open(self, fp, mode=None):
        if mode is None: mode='w'
        self.file = open(fp, mode)
    def write(self, msg, is_terminal=1, is_file=1):
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
def img_save(imgs, to_path):
    cv2.imwrite(to_path, cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR))
def adjust_learning_rate(args, optimizer, cur_iter):
    if args.lr_decay_iter < cur_iter:
        for g in optimizer.param_groups:
            lr -= args.lr / (args.n_iters - args.lr_decay_iter)
            g["lr"] = lr
def denorm(x):
    return x * 0.5 + 0.5
def make_heatmap(x):
    min_, max_ = np.min(x), np.max(x)
    cam_img = (x - min_) / (max_ - min_)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img
    



    
