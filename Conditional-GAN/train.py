import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torchvision.utils import save_image

class args:
    batch_size = 64
    num_epochs = 200000
    
    n_latent = 100
    n_classes = 10
    img_shape = [1,28,28]
    
    lr_G = 0.0004
    lr_D = 0.0002
    betas_G = (0.5, 0.999)
    betas_D = (0.5, 0.999)
    gpus = "0"
    
    
    n_col = 8
    save_img_dir = "./gene_images"
    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_one_hot(args, label, device):
    one_hot = torch.zeros(len(label), args.n_classes)
    one_hot[range(len(label)), label] = 1
    return one_hot.to(device)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(args.n_classes, args.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(args.n_latent + args.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(args.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *args.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(args.n_classes, args.n_classes)

        self.model = nn.Sequential(
            nn.Linear(args.n_classes + int(np.prod(args.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

transform = T.Compose([
    T.ToTensor()
])
train_dataset = MNIST(root="/data", download = True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size = args.batch_size, drop_last=True, shuffle=True)
generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G, betas=args.betas_G)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr_D, betas=args.betas_D)
criterion_BCE = nn.MSELoss()

one_label = Variable(torch.ones(args.batch_size), requires_grad=False).to(device)
zero_label = Variable(torch.zeros(args.batch_size), requires_grad=False).to(device)
for epoch in range(args.num_epochs):
    G_loss_sum = 0
    D_loss_sum = 0
    train_loop = tqdm(train_loader, total = len(train_loader), leave = False)
    for img, label in train_loop:
        img = img.to(device)
        label = label.to(device)
        #one_hot_label = to_one_hot(args, label, device)
        gene_label = Variable(torch.LongTensor(np.random.randint(0, args.n_classes, args.batch_size))).to(device)
        #gene_one_hot_label = to_one_hot(args, gene_label, device)
        latent_z = Variable(torch.randn(args.batch_size, args.n_latent)).to(device)
        gene_img = generator(latent_z, label)
        # training D
        real_logit = discriminator(img, label).squeeze()
        gene_logit = discriminator(gene_img.detach(), gene_label).squeeze()
        D_loss = (criterion_BCE(real_logit, one_label) + criterion_BCE(gene_logit, zero_label)) / 2
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()        
        
        # training G
        gene_logit = discriminator(gene_img, gene_label).squeeze()
        G_loss = criterion_BCE(gene_logit, one_label)
        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()
        
        D_loss_sum += D_loss.item()
        G_loss_sum += G_loss.item()
        
    gene_samples = None
    for col in range(args.n_col):
        latent_z = Variable(torch.randn(args.n_classes, args.n_latent)).to(device)
        label = Variable(torch.arange(args.n_classes)).to(device)
        one_hot_label = to_one_hot(args, label, device)
        gene_img = generator(latent_z, label)
        
        row_imgs = gene_img[0]
        for img in gene_img[1:]:
            row_imgs = torch.cat([row_imgs, img], axis=2)
            
        if col == 0:
            gene_samples = row_imgs
        else:
            gene_samples = torch.cat([gene_samples, row_imgs], axis = 1)
            
    if not os.path.isdir(args.save_img_dir):
        os.makedirs(args.save_img_dir)
    save_image(gene_samples, f"{args.save_img_dir}/{epoch}.jpg")
    print(f"G loss : {G_loss_sum / len(train_loader)}, D loss : {D_loss_sum / len(train_loader)}")
        
        