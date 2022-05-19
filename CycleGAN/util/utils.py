import sys
from math import exp
import torch
import torch.nn.functional as F
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

if __name__ == "__main__":
    for i in range(100):
        sample_tensor1 = torch.randn((4,3,128,128)).clip(0,1).cuda(0)
        sample_tensor2 = torch.randn((4,3,128,128)).clip(0,1).cuda(0)

        ms_ = MSSSIM()
        print(ms_(sample_tensor1, sample_tensor2))
        import time
        time.sleep(2)