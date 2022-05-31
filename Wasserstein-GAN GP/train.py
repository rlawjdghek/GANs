from models.models import Generator, Discriminator
from loss.loss import compute_gp_loss

import torchvision
import torchvision.transforms as T
import torch

import os
import numpy as np
from tqdm import tqdm

class args:
    n_epochs = 200000
    batch_size = 256
    latent_dim = 10
    input_shape = [1,28,28]
    
    G_dim_list = [128, 256, 512, 1024]
    D_per_G = 5
    
    lambda_gp = 10
    lr = 0.0002
    betas=(0.5, 0.999)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

transform = T.Compose([
    T.RandomHorizontalFlip(0.5),
    T.ToTensor()
])
train_dataset = torchvision.datasets.MNIST(root = "/jupyterdata/", download=True, transform=transform, train=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, drop_last=True)

generator = Generator(latent_dim=args.latent_dim,
                      dim_list=args.G_dim_list,
                      input_shape=args.input_shape).to(device)
discriminator = Discriminator(input_shape=args.input_shape).to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr = args.lr, betas=args.betas) 
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = args.lr, betas=args.betas)

for epoch in range(args.n_epochs):
    total_D_loss = 0
    total_G_loss = 0
    train_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave = False)
    for i, (img, label) in train_loop:
        latent_z = torch.randn((args.batch_size, args.latent_dim)).to(device)
        img = img.to(device)
        label = label.to(device)
        
        gene_img = generator(latent_z)
        # train discriminator
        real_logit = discriminator(img)
        gene_logit = discriminator(gene_img.detach())
        D_gp_loss = compute_gp_loss(discriminator, img.data, gene_img.data, device)
        D_loss = -torch.mean(real_logit) + torch.mean(gene_logit) + args.lambda_gp * D_gp_loss  # real은 높게, gene는 낮게해야 되므로 
        # real에 음수를 붙여 높은 값이 나올수록 loss는 커진다.
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()
        
        # train generator
        if i % args.D_per_G == 0:
            gene_logit = discriminator(gene_img)
            G_loss = -torch.mean(gene_logit)
            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()
        total_D_loss += D_loss.data
        total_G_loss += G_loss.data
        train_loop.set_description(f"Epoch : [{epoch}/{args.n_epochs}]")
    print(f"D loss : {total_D_loss / len(train_loader)}, G loss : {total_G_loss / len(train_loader)}")
    if not os.path.isdir("./generated_image"):
        os.makedirs("./generated_image")
    torchvision.utils.save_image(gene_img[:16], "./generated_image/{}.png".format(epoch), nrow=4)
            
            
        