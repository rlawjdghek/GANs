import os
from os.path import join as opj
import random
import numpy as np

from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.pgm', '.PGM',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', 
    '.txt', '.json'
]

class BaseDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
    def name(self):
        return "BaseDataset"
def is_image_file(f):
    return any(f.endswith(extension) for extension in IMG_EXTENSIONS)
def make_grouped_dataset(dir):
    images = []
    fnames = sorted(os.walk(dir))
    for fname in sorted(fnames):
        paths = []
        root = fname[0]
        for f in sorted(fname[2]):
            if is_image_file(f):
                paths.append(opj(root, f))
        if len(paths) > 0:
            images.append(paths)
    return images
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
def get_img_params():
    flip = random.random() > 0.5
    return {"flip": flip}
def get_transforms(args, params, is_train=True):
    T_lst = []
    T_lst.append(T.Resize((args.img_size, args.img_size), interpolation=Image.BICUBIC))
    if is_train and args.use_flip:
        T_lst.append(T.Lambda(lambda img: _flip(img, params["flip"])))
    T_lst.append(T.ToTensor())
    if args.input_ch == 1:
        T_lst.append(T.Normalize((0.5), (0.5)))
    elif args.input_ch == 3:
        T_lst.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return T.Compose(T_lst)
def get_video_params(args, n_frames_total, cur_seq_len, idx, is_train):
    tG = args.n_frames_G
    if is_train:
        n_frames_total = min(n_frames_total, cur_seq_len - tG + 1)
        n_frames_total = n_frames_total + tG - 1
        max_t_step = min(args.max_t_step, (cur_seq_len-1)//(n_frames_total-1))
        t_step = np.random.randint(max_t_step)+1
        offset_max = max(1, cur_seq_len - (n_frames_total-1)*t_step)
        start_idx = np.random.randint(offset_max)
    else:
        n_frames_total = tG
        start_idx = idx
        t_step = 1
    return n_frames_total, start_idx, t_step






    