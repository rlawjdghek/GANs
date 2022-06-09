from glob import glob
from os.path import join as opj
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as T

from .base_dataset import BaseDataset, get_transform

class UnAlignedDataset(BaseDataset):
    def __init__(self, args, is_train):
        super().__init__(args)
        self.is_train = is_train
        self.data_name = args.data_name
        self.torv = "train" if self.is_train else "valid"
        if "VFP290K" in self.data_name: interval = 5
        else: interval = 1
        self.A_paths = glob(opj(args.data_root_dir, self.data_name, self.torv, "A/*.jpg"))[::interval]
        self.A_paths += glob(opj(args.data_root_dir, self.data_name, self.torv, "A/*.png"))[::interval]
        self.B_paths = glob(opj(args.data_root_dir, self.data_name, self.torv, "B/*.jpg"))
        self.B_paths += glob(opj(args.data_root_dir, self.data_name, self.torv, "B/*.png"))
        self.n_A = len(self.A_paths)
        self.n_B = len(self.B_paths)
        self.transform_A = get_transform(args, is_train=is_train, grayscale=args.in_ch==1, use_crop=args.use_crop_A)
        self.transform_B = get_transform(args, is_train=is_train, grayscale=args.in_ch==1, use_crop=args.use_crop_B)
    def name(self):
        return "UnAlignedDataset"
    def __len__(self):
        return max(self.n_A, self.n_B)
    def __getitem__(self, idx):
        A_path = self.A_paths[idx % self.n_A]
        B_path = self.B_paths[idx % self.n_B]
        raw_A = Image.open(A_path).convert("RGB")
        raw_B = Image.open(B_path).convert("RGB")
        img_A = self.transform_A(raw_A)
        img_B = self.transform_B(raw_B)
        return {"img_A": img_A, "img_B": img_B}     
class TestDataset_single(BaseDataset):  # 스타일 이미지 다양하게 주고 싶을떄
    def __init__(self, args, test_data_dir):
        self.img_paths = glob(opj(test_data_dir, "*.jpg"))
        self.img_paths += glob(opj(test_data_dir, "*.png"))
        self.n_paths = len(self.img_paths)
        self.paths_num_check()
        self.transforms = T.Compose([
            T.Resize((args.img_H, args.img_W)),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
    def name(self):
        return "TestDataset single"
    def paths_num_check(self):
        print(f"[Test] # of imgs : {self.n_paths}")
    def __len__(self):
        return self.n_paths
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = self.transforms(Image.open(path).convert("RGB"))
        return {"img" : img}
class TestDataset_pair(BaseDataset):  # 일대일 테스트
    def __init__(self, args, test_data_dir):
        self.A_paths = glob(opj(test_data_dir, "A/*.jpg"))
        self.A_paths += glob(opj(test_data_dir, "A/*.png"))
        self.B_paths = glob(opj(test_data_dir, "B/*.jpg"))
        self.B_paths += glob(opj(test_data_dir, "B/*.png"))
        self.n_A = len(self.A_paths)
        self.n_B = len(self.B_paths)
        self.paths_num_check()
        self.transforms = T.Compose([
            T.Resize((args.img_H, args.img_W)),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
    def name(self):
        return "TestDataset pair"
    def paths_num_check(self):
        print(f"[Test] # of A : {len(self.A_paths)}, # of B : {len(self.B_paths)}")
    def __len__(self):
        return max(self.n_A, self.n_B)
    def __getitem__(self, idx):
        A_path = self.A_paths[idx % self.n_A]
        B_path = self.B_paths[idx % self.n_B]
        img_A = self.transforms(Image.open(A_path).convert("RGB"))
        img_B = self.transforms(Image.open(B_path).convert("RGB"))
        return {"img_A": img_A, "img_B": img_B}

        
        
