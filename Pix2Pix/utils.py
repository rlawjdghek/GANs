import numpy as np
import matplotlib.pyplot as plt
import os

import torch

def save_figure(real_imgs : torch.FloatTensor, gene_imgs : torch.FloatTensor, sketch : torch.FloatTensor, to_dir : str, epoch):
    real_imgs = real_imgs.numpy()
    gene_imgs = gene_imgs.numpy()
    sketch = sketch.numpy()

    img_skt = np.concatenate([real_imgs, gene_imgs, sketch], axis = 3)
    img_skt = np.transpose(img_skt, (0, 2, 3, 1))
    img_skt = (img_skt+1) / 2

    to_dir = os.path.join(to_dir, str(epoch))
    if not os.path.isdir(to_dir):
        os.makedirs(to_dir)

    for i, img in enumerate(img_skt):
        plt.imsave(os.path.join(to_dir, "{}.png".format(i)), img)
