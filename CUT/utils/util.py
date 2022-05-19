import sys

import numpy as np
import cv2
import torch
from torchvision.utils import make_grid

class Logger(object):
    def __init__(self, local_rank):
        self.terminal = sys.stdout
        self.file = None
        self.local_rank = local_rank

    def open(self, fp, mode=None):
        if mode is None: mode = 'w'
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
def print_args(args, logger=None):
  for k, v in vars(args).items():
      if logger is not None:
          logger.write('{:25s}: {}\n'.format(k, v))
      else:
          print('{:25s}: {}'.format(k, v))
def _psnr(output, target):
    '''
    :param output: tensor with 0 ~ 1 float [B x C x H x W]
    :param target: tensor with 0 ~ 1 float [B x C x H x W]
    '''
    assert output.shape == target.shape, "input image shape must have same shape"
    output = torch.clip(output, 0, 1)
    target = torch.clip(target, 0, 1)
    mse = torch.mean((output - target) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
def img_save(real_A, gene_B, real_B, to_path, local_rank=0):  # img : torch.tensor
    real_A = make_grid(real_A[:4], nrow=2, padding=0)
    gene_B = make_grid(gene_B[:4], nrow=2, padding=0)
    real_B = make_grid(real_B[:4], nrow=2, padding=0)
    img = torch.cat([real_A, gene_B, real_B], dim=2)  # [C x H x W]
    img = img.permute(1,2,0).numpy()
    img = (img+1)/2
    img = np.clip(img, 0, 1)
    img = np.uint8(img*255.0)
    if local_rank==0:
        cv2.imwrite(to_path, img)
    