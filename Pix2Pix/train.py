import sys
import numpy as np
import os
from glob import glob
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import style
style.use("dark_background")

import torch
import torchvision
from torchvision import transforms

from model import *
from utils import save_figure

class Logger(object):  
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None 
    
    def open(self, file_path, mode=None):
        if mode is None: mode = 'w'
        self.file = open(file_path, mode)
    
    def write(self, message, is_terminal=1, is_file=1):
        if "\r" in message: is_file=0
        if is_terminal:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file:
            self.file.write(message)
            self.file.flush()        

# arguments & logger
class args:
    batch_size = 32
    learning_rate = 2e-4
    b1 = 0.5
    b2 = 0.999
    epochs = 1000
    H = 256
    W = 256
    alpha = 100
    log_dir = "./log"
    log_name = "log.txt"
    save_dir = "./save_model"
    save_paths = ["./{}/generator_1.pth".format(save_dir), "./{}/discriminator_1.pth".format(save_dir)]
    figure_dir = "./gene_figure"
    
class Logger(object):  
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None 
    
    def open(self, file_path, mode=None):
        if mode is None: mode = 'w'
        self.file = open(file_path, mode)
    
    def write(self, message, is_terminal=1, is_file=1):
        if "\r" in message: is_file=0
        if is_terminal:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file:
            self.file.write(message)
            self.file.flush()
class facades_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode="train"):
        self.paths = glob(os.path.join(data_dir, mode, "*"))
        print("{} => num_imgs : {}".format(mode, len(self.paths)))
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_COLOR) / 255.0
        h, w = img.shape[:2]
        img = np.transpose(img, (2, 0 ,1)) # [H x W x C] => [C x H x W]
        img = (img-0.5) / 0.5  # normalization
        real = img[:, :, w//2:w]
        sketch = img[:, :, :w//2]
        if np.random.random() < 0.5:  # sketch flip 하면 real 도 flip해야 하므로 이렇게 해야된다. transpose로 하면 안 맞을 수 있다. 
            real = real[:, :, ::-1].copy()
            sketch = sketch[:, :, ::-1].copy()
        real = torch.FloatTensor(real)
        sketch = torch.FloatTensor(sketch)
        return real, sketch

train_dataset = facades_dataset("../data/edges2shoes/edges2shoes/", mode="train")
valid_dataset = facades_dataset("../data/edges2shoes/edges2shoes/", mode="val")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
real_label = torch.autograd.Variable(torch.ones((args.batch_size, 1, args.H // 16, args.W // 16)), requires_grad=False).to(device)
gene_label = torch.autograd.Variable(torch.zeros((args.batch_size, 1, args.H // 16, args.W // 16)), requires_grad=False).to(device)

generator = GeneratorUNet().to(device)
discriminator = Discriminator().to(device)
if os.path.isfile(args.save_paths[0]):
    generator.load_state_dict(torch.load(args.save_paths[0]))
    print("generator load success!!!!")
if os.path.isfile(args.save_paths[1]) :
    discriminator.load_state_dict(torch.load(args.save_paths[1]))
    print("discriminator load success!!!!")
    
optimizer_G = torch.optim.Adam(generator.parameters(), lr = args.learning_rate, betas=[args.b1, args.b2])
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = args.learning_rate, betas=[args.b1, args.b2])
criterion_MSE = torch.nn.MSELoss()
criterion_L1 = torch.nn.L1Loss()
    


logger = Logger()
if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)
    print("make directory : {}".format(args.log_dir))
logger.open(os.path.join(args.log_dir, args.log_name))

for epoch in range(args.epochs):
    print("epoch : {}".format(epoch))
    total_G_loss = torch.FloatTensor([0.])
    total_D_loss = torch.FloatTensor([0.])   
    
    for i, (real_img, sketch) in enumerate(train_loader):
        real_img = real_img.to(device)
        sketch = sketch.to(device)
        
        gene_img = generator(sketch)
    
        # train discriminator
        real_preds = discriminator(real_img, sketch)
        gene_preds = discriminator(gene_img.detach(), sketch)
        D_real_loss = criterion_L1(real_preds, real_label)
        D_gene_loss = criterion_L1(gene_preds, gene_label)
        D_loss = (D_real_loss + D_gene_loss) / 2
        
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()
        
        # train generator
        gene_preds = discriminator(gene_img, sketch)
        G_pixel_loss = criterion_MSE(real_img, gene_img)
        G_adv_loss = criterion_L1(gene_preds, real_label)
        G_loss = G_adv_loss + args.alpha * G_pixel_loss
        
        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()
        
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
            print("make directory : {}".format(args.save_dir))
        torch.save(generator.state_dict(), "./save_model/generator_{}.pth".format(epoch))
        torch.save(discriminator.state_dict(), "./save_model/discriminator_{}.pth".format(epoch))
        logger.write("\r Batch : [{}/{}] => G loss : {}, D loss : {}".format(i, len(train_loader), G_loss, D_loss))
        
        total_G_loss += G_loss.cpu()
        total_D_loss += D_loss.cpu()
    print("G loss : {}, D loss : {}".format(G_loss / i, D_loss / i))
    save_figure(real_img.detach().cpu(), gene_img.detach().cpu(), sketch.detach().cpu(), args.figure_dir, epoch)