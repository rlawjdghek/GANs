import random
from abc import ABC, abstractmethod
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


def _scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)
def _crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1+tw, y1+th))
    return img
def _make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)
def _flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
def get_params(args, size):
    w, h = size
    new_h = h
    new_w = w
    if args.resize and args.crop:
        new_h = new_w = args.size
    elif args.resize and args.crop:
        new_w = args.size
        new_h = args.size * h // w

    x = random.randint(0, np.maximum(0, new_w - args.crop_size))
    y = random.randint(0, np.maximum(0, new_h - args.crop_size))
    flip = random.random() > 0.5
    return {"crop_pos": (x, y), "flip": flip}
def get_transform(args, params=None, grayscale=False, method=T.InterpolationMode.BICUBIC, normalize=True, is_train=True):
    T_lst = []
    if grayscale:
        T_lst.append(T.Grayscale(1))
    if is_train and args.resize:
        if args.resize:
            img_size = [args.size, args.size]
            T_lst.append(T.Resize(img_size, method))
        elif args.scale_width:
            T_lst.append(T.Lambda(lambda img: _scale_width(img, args.size, method)))
        else:
            assert 1 == 0, "need to resize or scale width at least!!!"

    if is_train and args.crop:
        if params is None:
            T_lst.append(T.RandomCrop(args.crop_size))
        else:
            T_lst.append(T.Lambda(lambda img: _crop(img, params["crop_pos"], args.crop_size)))

    if not (args.resize or args.scale_width or args.crop) :
        T_lst.append(T.Lambda(lambda img: _make_power_2(img, base=4, method=method)))

    if is_train and args.flip:
        if params is None:
            T_lst.append(T.RandomHorizontalFlip())
        elif params["flip"]:
            T_lst.append(T.Lambda(lambda img: _flip(img, params["flip"])))

    if not is_train:
        img_size = [args.crop_size, args.crop_size]
        T_lst.append(T.Resize(img_size, method))

    T_lst.append(T.ToTensor())
    if normalize:
        if grayscale: T_lst.append(T.Normalize((0.5), (0.5)))
        else: T_lst.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return T.Compose(T_lst)
class BaseDataset(Dataset, ABC):
    def __init__(self):
        super().__init__()
    def name(self):
        return "BaseDataset"
    @abstractmethod
    def paths_num_check(self):
        pass

