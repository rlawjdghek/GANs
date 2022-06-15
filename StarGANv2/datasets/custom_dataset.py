import os
from os.path import join as opj
from glob import glob
import random
import numpy as np

from PIL import Image
import torchvision.transforms as T

from .base_dataset import BaseDataset

class AFHQDataset(BaseDataset):
    def __init__(self, args, transform):
        super().__init__(args)
        self.args = args
        domain_names = sorted(os.listdir(opj(args.data_root_dir, args.data_name, "train")))
        self.n_domains = len(domain_names)
        self.img_paths = []
        self.labels = []
        for label, domain_name in enumerate(domain_names):
            paths = glob(opj(args.data_root_dir, args.data_name, "train", domain_name, "*"))
            self.img_paths += paths
            self.labels += [label for _ in range(len(paths))]
        assert len(self.img_paths) == len(self.labels)
        # shuffle for DDP
        self.img_paths = np.asarray(self.img_paths)
        self.labels = np.asarray(self.labels)
        _idx = np.arange(len(self.img_paths))
        np.random.shuffle(_idx)
        self.img_paths = self.img_paths[_idx]
        self.labels = self.labels[_idx]       
        self.n_imgs = len(self.img_paths)
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))
        return {"img": img, "label": label}
class AFHQRefDataset(BaseDataset):
    def __init__(self, args, transform):
        super().__init__(args)
        self.args = args
        domain_names = sorted(os.listdir(opj(args.data_root_dir, args.data_name, "train")))
        self.img_paths, self.img_paths2, self.labels = [], [], []
        for idx, domain in enumerate(sorted(domain_names)):
            paths = glob(opj(args.data_root_dir, args.data_name, "train", domain, "*"))
            self.img_paths += paths
            self.img_paths2 += random.sample(paths, len(paths))
            self.labels += [idx for _ in range(len(paths))]
        assert len(self.img_paths) == len(self.img_paths2) == len(self.labels)
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_path2 = self.img_paths2[idx]
        label = self.labels[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))
        img2 = self.transform(Image.open(img_path2).convert("RGB"))
        return {"img1": img, "img2": img2, "label": label}
class SingleDataset(BaseDataset):
    def __init__(self, data_dir, transform):
        super().__init__(None)
        self.img_paths = glob(opj(data_dir, "*"))
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))
        return {"img": img}
        


