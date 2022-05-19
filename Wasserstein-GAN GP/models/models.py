import numpy as np

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, dim_list, input_shape = [1,28,28]):
        super(Generator, self).__init__()
        self.input_shape = input_shape
        blocks = [nn.Linear(latent_dim, dim_list[0]), nn.LeakyReLU(0.2)]
        for in_ch, out_ch in zip(dim_list[:-1], dim_list[1:]):
            blocks.extend(self._basic_block(in_ch, out_ch))
        blocks.append(nn.Linear(dim_list[-1], int(np.prod(input_shape))))
        blocks.append(nn.Tanh())
        self.generator = nn.Sequential(*blocks)
            
        
        
    def _basic_block(self, in_channels, out_channels, bn=True):
        block = [nn.Linear(in_channels, out_channels)]
        if bn:
            block.append(nn.BatchNorm1d(out_channels))
        block.append(nn.LeakyReLU(0.2))
        return block

    def forward(self, z):
        output = self.generator(z)  # output => [BATCH_SIZE x CHANNELS x W x H]
        return output.reshape(output.shape[0], *self.input_shape)
        
        
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, input_img):
        input_flat = input_img.reshape(input_img.shape[0], -1)
        return self.discriminator(input_flat)  # return => [BATCH_SIZE x 1]
    
if __name__ == "__main__":
    sample = torch.randn((4,10))
    generator = Generator(10, [128,256,512,1024], [1,28,28])
    discrimintaor = Discriminator()
    gene_output = generator(sample)
    dis_output = discrimintaor(gene_output)
    print(f"gene output : {gene_output.shape}, dis output : {dis_output.shape}")