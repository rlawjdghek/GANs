import random
from abc import ABC, abstractmethod
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

def _scale_height(img, target_height, method=Image.BICUBIC):
    ow, oh = img.size
    if oh == target_height:
        return img
    h = target_height
    w = int(target_height * ow / oh)
    return img.resize((w, h), method)
def get_transform(args, is_train=True, grayscale=False, use_crop=True):
    T_lst = []
    if grayscale:
        T_lst.append(T.Grayscale(1))
    #### train ####
    if is_train:
        if use_crop:  # crop 하면 resize해서 먼저 이미지 키워두고 random crop
            if args.resize_type == "resize":
                T_lst.append(T.Resize((args.resize_H, args.resize_W)))
            elif args.resize_type == "scale_height":
                T_lst.append(T.Lambda(lambda img: _scale_height(img, args.img_H*2)))
            T_lst.append(T.RandomCrop(args.img_H, args.img_W))
        else: T_lst.append(T.Resize((args.img_H, args.img_W)))
    #### valid ####
    if not is_train:
        T_lst.append(T.Resize((args.resize_H, args.resize_W)))
    T_lst.append(T.ToTensor())
    T_lst.append(T.Normalize(0.5, 0.5))
    return T.Compose(T_lst)    
class BaseDataset(Dataset, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
    @abstractmethod
    def name(self):
        return "BaseDataset"