from models import Generator, Discriminator
from datasets import CelebA_dataset
from loss.gradient_penalty import compute_gradient_penalty
from utils.utils import Logger, make_samples

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision
from torch.utils.data import DataLoader

class args:
    # dataset
    num_workers=0
    batch_size = 16
    data_root_dir = "/jupyterdata/CelebA/"
    H = 128
    W = 128
    # hyperparameters
    epochs = 10000
    betas = (0.5, 0.999)
    lr = 0.0002
    dis_per_gene = 5  # 논문에서 G 1번에 D 5번 훈련했다. 
    lr_decay_step_size = 10
    lr_decay_gamma = 0.1
    in_ch = 3
    out_ch = 3
    n_domain = 5
    attributes = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
    lambda_cls = 1
    lambda_rec = 10
    lambda_gp = 10
    
    
    # generator
    gene_n_down = 3
    gene_n_bottle = 6
    gene_n_up = 2
    gene_n_cur_dim = 64
    
    # discriminator
    dis_n_hidden = 5
    dis_n_cur_dim = 64
    
    img_save_epoch=10
    changes = [
    [[0,1],[1,0],[2,0],[3,1],[4,1]],  # 검은 머리색 남자, 젊음
    [[0,0],[1,1],[2,0],[3,1],[4,1]],  # 금발 머리색 남자, 젊음
    [[0,0],[1,0],[2,1],[3,1],[4,1]],  # 갈색 머리색 남자, 젊음
    [[3,0],[4,0]],  # 기존 머리색 여자, 늙음
    [[3,0],[4,1]]  # 기본 머리색 여자, 젊음
    ]
    
logger=Logger()
logger.open("./log.txt")

transform = T.Compose([
    T.Resize((args.H, args.W)),
    T.RandomHorizontalFlip(0.5), 
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

train_dataset = CelebA_dataset(root_dir=args.data_root_dir, mode="train", transform=transform, attributes=args.attributes)
test_dataset = CelebA_dataset(root_dir=args.data_root_dir, mode="test", transform=transform, attributes=args.attributes)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(args.in_ch, args.out_ch, args.n_domain, args.gene_n_down, args.gene_n_bottle, args.gene_n_up, args.gene_n_cur_dim).to(device)
discriminator = Discriminator(args.in_ch, args.n_domain, args.dis_n_hidden, args.dis_n_cur_dim).to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=args.betas)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=args.betas)
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=args.lr_decay_step_size, gamma=args.lr_decay_gamma)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=args.lr_decay_step_size, gamma=args.lr_decay_gamma)
criterion_BCElogit = torch.nn.BCEWithLogitsLoss()
criterion_L1 = torch.nn.L1Loss()

iter_ = iter(test_loader)
if not os.path.isdir("./generated_images"):
    os.makedirs("./generated_images")
for epoch in range(args.epochs):
    sum_D_loss = 0
    sum_G_loss = 0
    train_loop = tqdm(enumerate(train_loader), total = len(train_loader), leave=False)
    for i, batch in train_loop:
        img = batch["img"]
        label = batch["label"]        
        img = img.to(device)
        label = label.to(device)
        
        random_c = torch.FloatTensor(np.random.randint(0, 2, (args.batch_size, args.n_domain))).to(device)
        gene_img = generator(img, random_c)
        
        # train discriminator
        pred_real, pred_label = discriminator(img)
        pred_fake, pred_random_c = discriminator(gene_img.detach())
        gradient_penalty = compute_gradient_penalty(discriminator, img.data, gene_img.data, device)
        
        D_adv_loss = -torch.mean(pred_real) + torch.mean(pred_fake) + args.lambda_gp * gradient_penalty  
        # 생각 잘 해보면, 이 값이 최소화 되기 위해서는 pred_real이 최대화 => 1
        # pred_fake가 최소화 => 0이 되어야 한다. 즉, BCE와 유사하다. 
        D_cls_loss = criterion_BCElogit(pred_label, label)
        D_loss = D_adv_loss + args.lambda_cls * D_cls_loss
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()
        
        if i % args.dis_per_gene ==0:  # D 5번에 G 1번
            # training generator
            gene_img = generator(img, random_c)
            recons_img = generator(gene_img, label)  # random c로 만들고, label을 넣어서 생성하면, label이 반영된 이미지가 나온다. 따라서 이것과 
            # 원래의 이미지로 로스를 구하면 재구성하는 로스가 된다.


            pred_fake, pred_label = discriminator(gene_img)
            G_adv_loss = -torch.mean(pred_fake)  # 이걸 최소화 하기 위해서는 fake를 넣었음에도 1이 되어야 한다. 
            G_cls_loss = criterion_BCElogit(pred_label, random_c)  # 생성하기 위해 random_c를 넣었으므로 random_c의 성질을 같은 이미지를 만들어야 한다.
            G_recons_loss = criterion_L1(img, recons_img)
            G_loss = G_adv_loss + args.lambda_cls * G_cls_loss + args.lambda_rec * G_recons_loss
            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()
            sum_D_loss += D_loss.item()
            sum_G_loss += G_loss.item()
            train_loop.set_description("batch : [{}/{}], D loss : {}, G loss : {}".format(i, len(train_loader), D_loss, G_loss))
        if i % args.img_save_epoch ==0:
            try: data = next(iter_)
            except : 
                iter_ = iter(test_loader)
                dawta = next(iter_)
                
            img = data["img"]
            label = data["label"]
            samples = make_samples(generator, img, label, args.n_domain, args.changes, device)
            torchvision.utils.save_image(samples.unsqueeze(0), f"./generated_images/{epoch}_{i}.png", normalize=True)
    scheduler_D.step()
    scheduler_G.step()
        
            
    if not os.path.isdir("./saved_model"):
        os.makedirs("./saved_model")
    torch.save(discriminator.state_dict(), "./saved_model/D_{}.pth".format(epoch))
    torch.save(generator.state_dict(), "./saved_model/G_{}.pth".format(epoch))
    logger.write(f"D_loss : {sum_D_loss / len(train_loader)}, G_loss : {sum_G_loss / len(train_loader)}\n")
    