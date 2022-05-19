import torch
import torch.nn as nn
import torchvision

class Feature_Extraction(nn.Module):  # 이 모델을 통과하면 가로 세로가 1/4 로 줄어든다.
    def __init__(self):
        super(Feature_Extraction, self).__init__()
        self.fe = nn.Sequential(*list(torchvision.models.vgg19(pretrained=True).features.children())[:18])
        
    def forward(self, img):
        return self.fe(img)
    
class Discriminator(nn.Module):  # without fully connect layer, output shape => [BATCH_SIZE x 1 x w//2**(n+1) x h//2**(n+1)]
    def __init__(self, channels, input_size):
        super(Discriminator, self).__init__()
        self.output_size = (input_size[0] // (len(channels) +1), input_size[1] // (len(channels) + 1))
        out_channels = channels[0]
        blocks = self._block(3, out_channels, first_bn=False)
        in_channels = out_channels
        for out_channels in channels:
            blocks.extend(self._block(in_channels, out_channels))
            in_channels = out_channels
        blocks.append(nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1))
        self.discriminator_model = nn.Sequential(*blocks)
        
    def _block(self, in_channels, out_channels, first_bn=True):
        block = [nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=1, padding=1)]
        if first_bn:
            block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.LeakyReLU(0.2))
        block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1))
        block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.LeakyReLU(0.2))
        return block
    
    def forward(self, img):
        return self.discriminator_model(img)
    
class res_block(nn.Module):
    def __init__(self, in_channels):
        super(res_block, self).__init__()
        out_channels = in_channels
        blocks = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]
        blocks.append(nn.BatchNorm2d(out_channels))
        blocks.append(nn.PReLU())
        blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        blocks.append(nn.BatchNorm2d(out_channels))
        self.res_block_model = nn.Sequential(*blocks)
    
    def forward(self, x):
        return x + self.res_block_model(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_res_blocks = 8):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(*[nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU()])
        res_blocks = []
        for i in range(n_res_blocks):
            res_blocks.append(res_block(64))
        self.resblocks = nn.Sequential(*res_blocks)
            
        self.conv2 = nn.Sequential(*[nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64)])
        
        self.upsample1 = nn.Sequential(*[
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PixelShuffle(upscale_factor=2),  # this module allocate of channels to height and width. In this case, channel goes to 
            #channel -> channel x 4 and height -> height x 2 and width -> width x 2
            nn.PReLU()
        ])
        self.upsample2 = nn.Sequential(*[
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU()
        ])
        
        self.conv3 = nn.Sequential(*[nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh()])
        
        
    def forward(self, img):
        x = self.conv1(img)
        x = self.resblocks(x)
        x = self.conv2(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.conv3(x)
        return x