from abc import ABC, abstractmethod
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision
from torch.optim import lr_scheduler

class L2RegLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.pow(x, 2)
        x = torch.mean(x)
        return x
class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()
    def forward(self, preds, is_target_real):
        loss_D = 0
        for pred in preds:
            output_pred = torch.sigmoid(pred) 
            if is_target_real:
                label = torch.ones(output_pred.shape).cuda(output_pred.get_device())
            else:
                label = torch.zeros(output_pred.shape).cuda(output_pred.get_device())
            loss_D += self.criterion(output_pred, label)
        return loss_D
class BCEHalfLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()
    def forward(self, pred):
        loss_D = 0
        for p in pred:
            output_p = torch.sigmoid(p)
            half_label = 0.5 * torch.ones((output_p.shape[0])).cuda(output_p.get_device())
            loss_D += self.criterion(output_p, half_label)
        return loss_D
def gaussian_weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1 and classname.find('Conv') == 0:
    m.weight.data.normal_(0.0, 0.02)
# def gaussian_weights_init(m, init_type="normal", init_gain=0.02):
#     classname = m.__class__.__name__
#     if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
#         if init_type == "normal":
#             init.normal_(m.weight.data, 0.0, init_gain)
#         elif init_type == "xavier":
#             init.xavier_normal_(m.weight.data, 0.0, init_gain)
#         elif init_type == "kaiming":
#             init.kaiming_normal_(m.weight.data, 0.0, init_gain)
#         elif init_type == "orthogonal":
#             init.orthogonal_(m.weight.data, 0.0, init_gain)
#         else:
#             raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#         if hasattr(m, "bias") and m.bias is not None:
#             init.constant_(m.bias.data, 0.0)
#     elif classname.find("BatchNorm2d") != -1:
#         init.normal_(m.weight.data, 1.0, init_gain)
#         init.constant_(m.bias.data, 0.0)
def get_scheduler(args, optimizer):
    if args.lr_scheduler == "linear":
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - args.linearlr_epochs) / float(args.n_epochs - args.linearlr_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_scheduler == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.steplr_step)
    elif args.lr_scheduler == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif args.lr_scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=0)
    else:
        raise NotImplementedError(f"learning rate scheduler {args.lr_scheduler} is not impleemented!!!"
                                  f"use ['linear', 'step', 'plateau', 'cosine']")
    return scheduler

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
class VGGLoss(nn.Module):
    def __init__(self, local_rank):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda(local_rank)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_gene_label=0.0):
        super().__init__()
        self.real_label = target_real_label
        self.gene_label = target_gene_label
        if use_lsgan:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCELoss()
    def forward(self, x, target_is_real):
        '''
        x는 여러개의 NLayerD로 이루어진 MultiscaleD에서 나옴. 리스트안에 리스트가 있으므로 마지막 prediction만 사용해야한다. 
        '''
        if isinstance(x[0], list):
            loss = 0
            for i in x:
                pred = i[-1]
                if target_is_real:
                    label = torch.zeros(pred.shape).fill_(self.real_label).cuda(pred.get_device())
                else:
                    label = torch.zeros(pred.shape).fill_(self.gene_label).cuda(pred.get_device())
                loss += self.criterion(pred, label)
        else:
            pred = x[-1]
            if target_is_real:
                label = torch.zeros(pred.shape).fill_(self.real_label).cuda(pred.get_device())
            else:
                label = torch.zeros(pred.shape).fill_(self.gene_label).cuda(pred.get_device())
            loss = self.criterion(pred, label)
        return loss
class ImagePool():
    '''
    default pool size = 50, 일단 50장을 모으고, 50장이 다 차면 1/2 확률로 기존의 50개에 있는것중 1개와 현재 있는 것을 바꿔서 내보낸다.
    배치 사이즈는 query 메소드에서 1이므로 for문은 1번 돈다.
    '''
    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []
    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images
class ImageMaskPool():
    '''
    default pool size = 50, 일단 50장을 모으고, 50장이 다 차면 1/2 확률로 기존의 50개에 있는것중 1개와 현재 있는 것을 바꿔서 내보낸다.
    배치 사이즈는 query 메소드에서 1이므로 for문은 1번 돈다.
    '''
    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []
            self.masks = []
    def query(self, images, masks):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images, masks
        return_images = []
        return_masks = []
        for image, mask in zip(images, masks):
            image = torch.unsqueeze(image.data, 0)
            mask = torch.unsqueeze(mask.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                self.masks.append(mask)
                return_images.append(image)
                return_masks.append(mask)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    tmp_mask = self.masks[random_id].clone()
                    self.images[random_id] = image
                    self.masks[random_id] = mask
                    return_images.append(tmp)
                    return_masks.append(tmp_mask)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
                    return_masks.append(mask)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return_masks = torch.cat(return_masks, 0)
        return return_images, return_masks
class BaseModel(ABC):
    def __init__(self, args):
        self.args = args
    @abstractmethod
    def set_input(self): pass
    @abstractmethod
    def train(self): pass
    @staticmethod
    def set_requires_grad(models, requires_grad=False):
        if not isinstance(models, list):
            models = [models]
        for model in models:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = requires_grad