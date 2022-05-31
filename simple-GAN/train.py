import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

BATCH_SIZE = 256
INPUT_SIZE = 28
LATENT_DIM = 100
INPUT_SHAPE = (1, INPUT_SIZE, INPUT_SIZE)
LEARNING_RATE = 1e-4
BETAS = (0.5, 0.999)
NUM_EPOCHS = 200

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
train_dataset = torchvision.datasets.MNIST(root = "../data", download = True, transform = transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, pin_memory = True, num_workers = 4, drop_last = True)

class Generator(nn.Module):
    def __init__(self, latent_dim, channel_list : list):
        super(Generator, self).__init__()
        blocks = [*self._basic_block(latent_dim, channel_list[0], bn = False)]
        for in_ch, out_ch in zip(channel_list[:-1], channel_list[1:]):
            blocks.extend(self._basic_block(in_ch, out_ch))
        self.generator = nn.Sequential(*blocks)
        self.generator.add_module("last linear", nn.Linear(channel_list[-1], int(np.prod(INPUT_SHAPE))))
        self.generator.add_module("sigmoid", nn.Tanh())
        
    def _basic_block(self, in_channels, out_channels, bn=True):
        block = []
        block.append(nn.Linear(in_channels, out_channels))
        if bn:
            block.append(nn.BatchNorm1d(out_channels))
        block.append(nn.LeakyReLU(0.2))
        return block

    def forward(self, latent_z):
        gene = self.generator(latent_z)  # gene => [BATHC_SIZE x LAST_CHANNEL]
        return gene.reshape(gene.shape[0], *INPUT_SHAPE)  # return => [BATCH_SIZE x CHANNELS x W x H]

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(INPUT_SHAPE)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1)       
        )
        
    def forward(self, input_img):
        input_flat = input_img.reshape(input_img.shape[0], -1)
        output = torch.sigmoid(self.model(input_flat))
        return output   


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(LATENT_DIM, [128,256,512, 1024]).to(device)
discriminator = Discriminator().to(device)

adversarial_loss = torch.nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr = LEARNING_RATE, betas=BETAS)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = LEARNING_RATE, betas=BETAS)


real_label = torch.autograd.Variable(torch.ones((BATCH_SIZE, 1)), requires_grad = False).to(device)
gene_label = torch.autograd.Variable(torch.zeros((BATCH_SIZE, 1)), requires_grad=False).to(device)

for epoch in range(NUM_EPOCHS):
    for i,(img, _) in enumerate(train_loader):
        optimizer_D.zero_grad()
        real_img = img.to(device)
        latent_z = nn.init.normal_(torch.zeros((BATCH_SIZE, LATENT_DIM))).to(device)
        gene_img = generator(latent_z)
        real_preds = discriminator(real_img)
        gene_preds = discriminator(gene_img.detach())
        dis_loss_real = adversarial_loss(real_preds, real_label)
        dis_loss_gene = adversarial_loss(gene_preds, gene_label)
        dis_loss = (dis_loss_real + dis_loss_gene) / 2
        dis_loss.backward()
        optimizer_D.step()
        
        optimizer_G.zero_grad()
        gene_preds = discriminator(gene_img)  # 여기서 generator 학습
        gen_loss = adversarial_loss(gene_preds, real_label)
        gen_loss.backward()
        optimizer_G.step()
        

        if i%300==0:
            print("EPOCH : [{}/{}], BATCH : [{}/{}], dis loss : {}, gen loss : {}".format(epoch, NUM_EPOCHS, i, len(train_loader), dis_loss, gen_loss))
    
    if epoch % 10 == 0:
        if not os.path.isdir("./result2"):
            print("make dir ./result2")
            os.makedirs("./result2")
        torchvision.utils.save_image(gene_img[:16], "./result2/{}.png".format(epoch), nrow = 5, normalize = True)