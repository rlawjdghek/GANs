import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import os

BATCH_SIZE = 256
INPUT_SIZE = 28
INPUT_SHAPE = (1,INPUT_SIZE, INPUT_SIZE)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
BETAS = (0.5, 0.999)
LATENT_DIM = 100
N_CRITIC = 1

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = torchvision.datasets.MNIST(root="../data", download = True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = True, drop_last = True, num_workers = 4, pin_memory = True, batch_size = BATCH_SIZE)


class Generator(nn.Module):
    def __init__(self, latent_dim, dim_list):
        super(Generator, self).__init__()
        blocks = [nn.Linear(latent_dim, dim_list[0]), nn.LeakyReLU(0.2)]
        for in_ch, out_ch in zip(dim_list[:-1], dim_list[1:]):
            blocks.extend(self._basic_block(in_ch, out_ch))
        blocks.append(nn.Linear(dim_list[-1], int(np.prod(INPUT_SHAPE))))
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
        return output.reshape(output.shape[0], *INPUT_SHAPE)
        
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(int(np.prod(INPUT_SHAPE)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, input_img):
        input_flat = input_img.reshape(input_img.shape[0], -1)
        return self.discriminator(input_flat)  # return => [BATCH_SIZE x 1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(LATENT_DIM, [128,256,512, 1024]).to(device)
discriminator = Discriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr = LEARNING_RATE, betas=BETAS)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = LEARNING_RATE, betas=BETAS)

for epoch in range(NUM_EPOCHS):
    for i, (img, _) in enumerate(train_loader):
        optimizer_D.zero_grad()
        real_img = img.to(device)
        latent_z = torch.nn.init.normal_(torch.zeros(BATCH_SIZE, LATENT_DIM)).to(device)
        gene_img = generator(latent_z)
        real_preds = discriminator(real_img)
        gene_preds = discriminator(gene_img.detach())
        
        loss_D = -torch.mean(real_preds) + torch.mean(gene_preds)
        loss_D.backward()
        optimizer_D.step()
        for p in discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)
            
        if i % N_CRITIC == 0:
            optimizer_G.zero_grad()
            gene_img = generator(latent_z)
            gene_preds = discriminator(gene_img)
            loss_G = -torch.mean(gene_preds)
            loss_G.backward()
            optimizer_G.step()
            
        if i % 100==0:
            print("EPOCH : [{}/{}], BATCH : [{}/{}], loss D : {}, loss G : {}".format(epoch, NUM_EPOCHS, i, len(train_loader), loss_D, loss_G))
            
    if epoch % 10 ==0:
        if not os.path.isdir("./result"):
            print("make ./result")
            os.makedirs("./result")
        torchvision.utils.save_image(gene_img[:16], "./result/epoch{:d}.png".format(epoch), nrow=4)