import os
from os.path import join as opj

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import torchvision
import torchvision.transforms as T

class args:
    #### dataset ####
    img_size = 32
    channels = 1
    
    #### training ####
    batch_size = 256
    n_epochs = 200
    lr = 2e-4
    latent_dim = 100
    
    lambda_gp = 10
    
    #### save & load ####
    save_img_dir = "./save_images"
os.makedirs(args.save_img_dir, exist_ok=True)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = args.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(args.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, args.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
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
            *discriminator_block(args.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = args.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
def compute_gradient_penalty(D, X):
    """Calculates the gradient penalty loss for DRAGAN"""
    # Random weight term for interpolation
    alpha = torch.FloatTensor(np.random.random(size=X.shape)).cuda()

    interpolates = alpha * X + ((1 - alpha) * (X + 0.5 * X.std() * torch.rand(X.size()).cuda()))
    interpolates = Variable(interpolates, requires_grad=True)

    d_interpolates = D(interpolates)

    fake = Variable(torch.FloatTensor(X.shape[0], 1).fill_(1.0).cuda(), requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = args.lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

G = Generator()
D = Discriminator()
G.apply(weights_init_normal)
D.apply(weights_init_normal)
G = G.cuda()
D = D.cuda()

criterion_adv =nn.BCELoss()

transform = T.Compose([
    T.Resize(args.img_size),
    T.ToTensor(),
    T.Normalize((0.5), (0.5))
])
train_dataset = torchvision.datasets.MNIST("./mnist", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

for epoch in range(args.n_epochs):
    print(epoch)
    for idx, (img, _) in enumerate(train_loader):
        img = img.cuda()
        one_label = torch.ones((img.shape[0], 1)).cuda()
        zero_label = torch.zeros((img.shape[0], 1)).cuda()
        
        optimizer_G.zero_grad()
        z = torch.normal(0 ,1, (img.shape[0], args.latent_dim)).cuda()
        gene_img = G(z)
        g_loss = criterion_adv(D(gene_img), one_label)
        
        g_loss.backward()
        optimizer_G.step()
        
        optimizer_D.zero_grad()
        real_loss = criterion_adv(D(img), one_label)
        fake_loss = criterion_adv(D(gene_img.detach()), zero_label)
        
        gp = compute_gradient_penalty(D, img.data)
        d_loss = (real_loss + fake_loss) / 2 + gp
        d_loss.backward()
        optimizer_D.step()
        
        
    
    to_path = opj(args.save_img_dir, f"[epoch-{epoch}].png")
    torchvision.utils.save_image(gene_img, to_path)
        
        