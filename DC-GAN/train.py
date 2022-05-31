import numpy as np
import math
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms


NUM_EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
LATENT_DIM = 100
IMG_SHAPE = (1,32,32) # 스트라이드 떄문에 2의 제곱
IMG_SIZE = 32
SAMPLE_INTERVAL = 500

transform = transforms.Compose([transforms.Resize((IMG_SHAPE[1], IMG_SHAPE[2])), transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])
train_dataset = torchvision.datasets.MNIST("../data", download = True, train = True, transform = transform)
train_loader = torch.utils.data.DataLoader(train_dataset, drop_last = True,batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

# 먼저 conv_name 과 batchnorm2d 가 오면 weight 초기화 이 표현 외워두기 나중에 Module의 apply함수와 잘 어울림
def weights_init_normal(layer):
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif layer_name.find("BatchNorm2d") != -1:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
        nn.init.constant_(layer.bias.data, 0.0)
        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # conv layer로 들어가기 위한 가로세로 길이 MNIST기준으로는 원본이 32 32인데 8 8을 처음 레이어로 넣는다. 그러고 나서 업샘플 2번으로 원본 형태로 맞춤
        self.init_size = IMG_SIZE // 4
        self.init_linear = nn.Linear(LATENT_DIM, 128 * (self.init_size ** 2))
        
        self.conv_layer = []
        # BatchNorm부터 시작하는것에 주의하자. 즉 여기에 들어가는 레이어의 채널은 128채널.
        self.conv_layer.append(nn.BatchNorm2d(128))
        self.conv_layer.append(nn.Upsample(scale_factor=2))
        self.conv_layer.append(nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1))
        self.conv_layer.append(nn.BatchNorm2d(128,0.8))
        self.conv_layer.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv_layer.append(nn.Upsample(scale_factor=2))
        self.conv_layer.append(nn.Conv2d(128, 64, kernel_size = 3, padding = 1, stride=1))
        self.conv_layer.append(nn.BatchNorm2d(64, 0.8))
        self.conv_layer.append(nn.LeakyReLU(0.2, inplace = True))
        self.conv_layer.append(nn.Conv2d(64, IMG_SHAPE[0], kernel_size=3, padding=1,stride=1))
        self.conv_layer.append(nn.Tanh())
        self.conv_layer = nn.Sequential(*self.conv_layer)
        
    def forward(self, z):
        latent = self.init_linear(z)
        latent = latent.reshape(BATCH_SIZE, 128, self.init_size, self.init_size)
        img = self.conv_layer(latent)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def block(in_channels, out_channels, bn = True):
            block = [nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_channels))
            return block
        
        self.model = nn.Sequential(
            *block(1, 16, bn = False), # 16
            *block(16,32),  # 8
            *block(32,64),  # 4
            *block(64,128)  # 2
        )
        
        
        ds_size = IMG_SHAPE[1] // (2 ** 4)
        self.adv_layer = nn.Sequential(nn.Linear(128* (ds_size ** 2), 1), nn.Sigmoid())
        
    def forward(self, img):
        output = self.model(img)
        output_flat = output.reshape(BATCH_SIZE, -1)
        output = self.adv_layer(output_flat)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator()
discriminator = Discriminator()
generator.to(device)
discriminator.to(device)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(generator.parameters(), lr = LEARNING_RATE, betas=(BETA1, BETA2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = LEARNING_RATE, betas=(BETA1, BETA2))
adversarial_loss = torch.nn.BCELoss()

real_label = Variable(torch.ones(BATCH_SIZE, 1), requires_grad=False).to(device)
fake_label = Variable(torch.zeros(BATCH_SIZE, 1), requires_grad=False).to(device)

for epoch in range(NUM_EPOCHS):
    for i, (img, _) in enumerate(train_loader):
        optimizer_D.zero_grad()
        
        z = nn.init.normal_(torch.zeros(BATCH_SIZE, LATENT_DIM)).to(device)
        
        fake_imgs = generator(z)
        real_imgs = img.to(device)
        
        fake_preds = discriminator(fake_imgs)
        real_preds = discriminator(real_imgs)
        
        dis_loss_fake = adversarial_loss(fake_preds, fake_label)
        dis_loss_real = adversarial_loss(real_preds, real_label)
        
        dis_loss = (dis_loss_fake + dis_loss_real) / 2
        dis_loss.backward()
        optimizer_D.step()
        
        
        optimizer_G.zero_grad()
        z = nn.init.normal_(torch.zeros(BATCH_SIZE, LATENT_DIM)).to(device)
        
        fake_imgs = generator(z)
        fake_preds = discriminator(fake_imgs)
        
        gen_loss = adversarial_loss(fake_preds, real_label)
        gen_loss.backward()
        optimizer_G.step()
        
        if i%100==0:
            print("EPOCH : [{}/{}], BATCH : [{}/{}], D loss : {}, G loss {}".format(epoch, NUM_EPOCHS, i, len(train_loader), dis_loss, gen_loss))
            
    if epoch % 10==0:
        if not os.path.isdir("./results"):
            os.makedirs("./results", exist_ok = True)
        torchvision.utils.save_image(fake_imgs[:25], "./results/{}.png".format(epoch), nrow=5, normalize=True)    
        
        
        