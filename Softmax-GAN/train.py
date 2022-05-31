import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T

class args:
    # hyperparams
    batch_size = 64
    n_epochs = 100
    input_shape = [1,28,28]  # in the case of mnist
    G_lr = 0.0002
    D_lr = 0.0002
    G_betas = (0.5, 0.999)
    D_betas = (0.5, 0.999)
    # model params
    n_latent = 100
    G_n_blocks = 2
    D_n_blocks = 3
    G_dim = 64
    D_dim = 64
    
    # device params
    gpus = "0"
    
device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.ToTensor()
])
train_dataset = torchvision.datasets.MNIST(root="/jupyterdata/", download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

# models
class Generator(nn.Module):
    def __init__(self, h, out_ch, n_latent, cur_dim, n_blocks = 2, k=3, s=1, p=1):
        super(Generator, self).__init__()
        self.init_dim = cur_dim
        self.init_size = h // (2**n_blocks)  # n_layer번 업샘플링 되므로 처음 채널 사이즈 조절
        self.linear = nn.Sequential(nn.Linear(n_latent, cur_dim * (self.init_size**2)))
        layers = []
        for _ in range(n_blocks):  # 한번 지날 수록 
            layers.append(self.basic_block(cur_dim, cur_dim*2, k, s, p))
            cur_dim *= 2
            
        layers.append(nn.Conv2d(cur_dim, out_ch, k,s,p))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)
            
    @staticmethod
    def basic_block(in_ch, out_ch, k=3, s=1, p=1):
        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, k, s, p))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Upsample(scale_factor=2))
        return nn.Sequential(*layers)
    
    def forward(self, input_):
        x = self.linear(input_)
        y = x.reshape(input_.shape[0], self.init_dim, self.init_size, self.init_size)
        return self.layers(y)
    
class Discriminator(nn.Module):
    def __init__(self, in_ch, cur_dim, n_blocks, h, k=3, s=2, p=1):
        super(Discriminator, self).__init__()
        feat_extract = []
        feat_extract.append(self.basic_block(in_ch, cur_dim))
        for _ in range(n_blocks-1):
            feat_extract.append(self.basic_block(cur_dim, cur_dim*2, k, s, p))
            cur_dim *= 2
        
        feat_extract.append(nn.AdaptiveAvgPool2d((1,1)))
        self.feat_extract = nn.Sequential(*feat_extract)
        self.linear = nn.Sequential(nn.Linear(cur_dim, 1))
        
    @staticmethod
    def basic_block(in_ch, out_ch, k=3, s=2, p=1):
        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, k, s, p))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, input_):
        x = self.feat_extract(input_)
        y = x.reshape(input_.shape[0], -1)
        return self.linear(y)
    
        
G = Generator(args.input_shape[1], args.input_shape[0], args.n_latent, args.G_dim, args.G_n_blocks).to(device)
D = Discriminator(args.input_shape[0], args.D_dim, args.D_n_blocks, args.input_shape[1]).to(device)

optimizer_G = torch.optim.Adam(G.parameters(), lr=args.G_lr, betas=args.G_betas)
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.D_lr, betas=args.D_betas)
for epoch in range(args.n_epochs):
    for img, target in train_loader:
        img = img.to(device)
        
        B = 1/(args.batch_size*2)
        
        latent_z = torch.FloatTensor(np.random.randn(args.batch_size, args.n_latent)).to(device)
        gene_img = G(latent_z)
        
        # training D
        real_logit = D(img)
        gene_logit = D(gene_img.detach())
        Z_B = torch.sum(torch.exp(-real_logit)) + torch.sum(torch.exp(-gene_logit))  # Z_B 는 모든 배치에 대하여 구하는 것.
        D_loss = torch.mean(real_logit) + torch.log(Z_B) 
        
        optimizer_D.zero_grad()
        D_loss.backward() 
        optimizer_D.step()
        
        # training G
        real_logit = D(img)
        gene_logit = D(gene_img)
        Z_B = torch.sum(torch.exp(-real_logit)) + torch.sum(torch.exp(-gene_logit))
        G_loss = torch.sum(real_logit) * B + torch.sum(gene_logit) * B + torch.log(Z_B)
        
        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()
    if not os.path.isdir("./generated_images"):
        os.makedirs("./generated_images")
    torchvision.utils.save_image(gene_img, "./generated_images/{}_2.jpg".format(epoch))
    print(f"D loss : {D_loss}, G loss : {G_loss}")