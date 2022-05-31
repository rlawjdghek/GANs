from os.path import join as opj
from glob import glob

from PIL import Image
import torchvision.transforms as T

from .base_dataset import BaseDataset

class UnAlignedDataset(BaseDataset):
    def __init__(self, args):
        self.args = args
        self.A_paths = sorted(glob(opj(args.data_root_dir, args.data_name, "trainA", "*")))
        self.B_paths = sorted(glob(opj(args.data_root_dir, args.data_name, "trainB", "*")))
        self.n_A = len(self.A_paths)
        self.n_B = len(self.B_paths)
        self.paths_num_check()
        self.transform = T.Compose([
            T.Resize((args.img_H, args.img_W)),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
    def name(self):
        return "UnAlignedDataset"
    def __len__(self):
        return max(self.n_A, self.n_B)
    def paths_num_check(self):
        print(f"[Train] # of A images : {len(self.A_paths)}, # of B images : {len(self.B_paths)}")
    def __getitem__(self, idx):
        A_path = self.A_paths[idx % self.n_A]
        B_path = self.B_paths[idx % self.n_B]
        img_A = self.transform(Image.open(A_path).convert("RGB"))
        img_B = self.transform(Image.open(B_path).convert("RGB"))
        return {"img_A": img_A, "img_B": img_B}

class TestDataset(BaseDataset):
    def __init__(self, args):
        self.args = args
        self.A_paths = sorted(glob(opj(args.data_root_dir, args.data_name, "testA", "*")))
        self.B_paths = sorted(glob(opj(args.data_root_dir, args.data_name, "testB", "*")))
        self.n_A = len(self.A_paths)
        self.n_B = len(self.B_paths)
        self.paths_num_check()
        self.transform = T.Compose([
            T.Resize((args.img_H, args.img_W)),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
    def name(self):
        return "UnAlignedDataset"
    def __len__(self):
        return max(self.n_A, self.n_B)
    def paths_num_check(self):
        print(f"[Test] # of A images : {len(self.A_paths)}, # of B images : {len(self.B_paths)}")
    def __getitem__(self, idx):
        A_path = self.A_paths[idx % self.n_A]
        B_path = self.B_paths[idx % self.n_B]
        img_A = self.transform(Image.open(A_path).convert("RGB"))
        img_B = self.transform(Image.open(B_path).convert("RGB"))
        return {"img_A": img_A, "img_B": img_B}
