import torch
import sys


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None
    
    def open(self, file_path, mode="w"):
        if mode is None: mode="w"
        self.file = open(file_path, mode)
        
    def write(self, message, is_terminal=1, is_file=1):
        if "\r" in message: is_file=0
        if is_terminal:
            self.terminal.write(message)
            self.terminal.flush()
            
        if is_file:
            self.file.write(message)
            self.file.flush()
            
    def flush(self):
        pass

def make_samples(generator, img, label, n_domain, changes, device):
    # changes를 주목해야 한다. 지금 예제에서는 5가지의 도메인 [검은 머리색, 금발 머리색, 갈색 머리색, 성별, 나이]로 되어있다.
    # 이제 5개를 모두 바꿔서 해볼 건데, 머리색은 3가지중 1개만 1로 되어있어야 한다. 성별, 나이는 1 또는 0
    samples = None
    img = img.to(device)
    label = label.to(device)
    for i in range(4):
        origin_img = img[i]  # [3 x H x W]
        origin_label = label[i]  # [n_domain]
        row_imgs = origin_img.repeat(n_domain, 1, 1, 1)  # [n_domain x 3 x H x W]
        row_labels = origin_label.repeat(n_domain, 1)  # [n_domain x n_domain]
        for ch_idx, change in enumerate(changes):
            for col, val in change:
                row_labels[ch_idx, col] = val
        gene_imgs = generator(row_imgs, row_labels)
        gene_imgs = torch.cat([img for img in gene_imgs.data], 2)  # [3 x H x (n_domain*W)]
        sample = torch.cat([origin_img.data, gene_imgs.data], 2)  # [3 x H x ((n_domain+1) * W)]
        samples = sample if samples is None else torch.cat([samples, sample], 1)
    return samples
       