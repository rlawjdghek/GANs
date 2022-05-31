from typing import List
import numpy as np
import os
from tqdm.auto import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

class args:
    gpus = "0"
    # dataset
    n_epochs = 200000
    batch_size = 64
    img_shape = [1,32,32]
    
    n_latent = 100
    
    G_lr = 0.0002
    G_betas = (0.5, 0.999)
    D_lr = 0.0002
    D_betas = (0.5, 0.999)
    
    gene_img_dir = Path("./generated_images")

if not os.path.isdir(args.gene_img_dir):
    os.makedirs(args.gene_img_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.ToTensor()
])
train_dataset = torchvision.datasets.MNIST(root="/data", download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)

def init_weight(layer):
    cls_name = layer.__class__.__name__
    if cls_name.find("Conv") != -1:
        torch.nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif cls_name.find("BatchNorm") != -1:
        torch.nn.init.normal_(layer.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(layer.bias.data, 0.0)
        

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = 28 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, 2),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

generator = Generator().to(device)
discriminator = Discriminator().to(device)
generator.apply(init_weight)
discriminator.apply(init_weight)                                         

criterion_MSE = torch.nn.MSELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr = args.G_lr, betas=args.G_betas)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = args.D_lr, betas=args.D_betas)

ones_label = Variable(torch.ones(args.batch_size), requires_grad=False).to(device)
zeros_label = Variable(torch.zeros(args.batch_size), requires_grad=False).to(device)
for epoch in range(args.n_epochs):
    train_loop = tqdm(train_loader, total=len(train_loader), desc="training", colour="blue", leave=False)
    G_loss_sum = 0
    D_loss_sum = 0
    for img, label in train_loop:
        img = img.to(device)
        label = label.to(device)
        latent_z = torch.randn(args.batch_size, args.n_latent).to(device)
        
        
        
        gene_img = generator(latent_z)
        # training D
        
        gene_logit = discriminator(gene_img.detach())
        real_logit = discriminator(img)
        D_loss = (criterion_MSE(gene_logit, zeros_label) + criterion_MSE(real_logit, ones_label)) / 2
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()    
        
        # training G
        gene_logit = discriminator(gene_img)
        G_loss = criterion_MSE(gene_logit, ones_label)
        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()
        
        D_loss_sum += D_loss.item()
        G_loss_sum += G_loss.item()
    print(f"D loss : {D_loss_sum/len(train_loader)}, G loss : {G_loss_sum/len(train_loader)}")
    save_image(gene_img, args.gene_img_dir / f"{epoch}.png")