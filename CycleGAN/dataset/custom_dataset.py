from glob import glob
from os.path import join as opj
from PIL import Image
import random

from .base_dataset import BaseDataset, get_transform

class UnAlignedDataset(BaseDataset):
    def __init__(self, args, logger, is_train):
        super().__init__()
        self.args = args
        self.logger = logger
        self.is_train = is_train
        self.data_name = args.data_name
        self.torv = "train" if self.is_train else "valid"
        self.ncct_paths = sorted(glob(opj(args.data_root_dir, self.data_name, self.torv, f"{self.args.AtoB_dir}/*.jpg")))
        self.cect_paths = sorted(glob(opj(args.data_root_dir, self.data_name, self.torv, f"{self.args.BtoA_dir}/*.jpg")))
        self.paths_num_check()
        self.n_ncct = len(self.ncct_paths)
        self.n_cect = len(self.cect_paths)
        self.ncct_T = get_transform(self.args, grayscale=(args.input_nc == 1), normalize=True, is_train=is_train)
        self.cect_T = get_transform(self.args, grayscale=(args.output_nc == 1), normalize=True, is_train=is_train)

    def paths_num_check(self):
        self.logger.write(f"[{self.torv}] # ncct: {len(self.ncct_paths)}, # cect: {len(self.cect_paths)}\n")

    def __len__(self):
        return max(self.n_ncct, self.n_cect)

    def __getitem__(self, idx):
        ncct_path = self.ncct_paths[idx % self.n_ncct]
        idx_cect = random.randint(0, self.n_cect-1)
        cect_path = self.cect_paths[idx_cect]
        ncct_img = self.ncct_T(Image.open(ncct_path).convert("RGB"))
        cect_img = self.cect_T(Image.open(cect_path).convert("RGB"))
        return {"ncct_img": ncct_img, "cect_img": cect_img}

class AlignedDataset(BaseDataset):
    def __init__(self, args, logger, is_train):
        super().__init__()
        self.args = args
        self.logger = logger
        self.is_train = is_train
        self.data_name = args.data_name
        self.torv = "train" if self.is_train else "valid"
        self.ncct_paths = sorted(glob(opj(args.data_root_dir, self.data_name, self.torv, "ncct/*.jpg")))
        self.cect_paths = sorted(glob(opj(args.data_root_dir, self.data_name, self.torv, "cect/*.jpg")))
        self.paths_num_check()
        self.n_ncct = len(self.ncct_paths)
        self.n_cect = len(self.cect_paths)
        self.ncct_T = get_transform(self.args, grayscale=(args.input_nc == 1), normalize=True, is_train=is_train)
        self.cect_T = get_transform(self.args, grayscale=(args.output_nc == 1), normalize=True, is_train=is_train)

    def paths_num_check(self):
        self.logger.write(f"[{self.torv}] # ncct: {len(self.ncct_paths)}, # cect: {len(self.cect_paths)}\n")

    def __len__(self):
        return max(self.n_ncct, self.n_cect)

    def __getitem__(self, idx):
        ncct_path = self.ncct_paths[idx % self.n_ncct]
        cect_path = self.cect_paths[idx % self.n_ncct]
        ncct_img = self.ncct_T(Image.open(ncct_path).convert("RGB"))
        cect_img = self.cect_T(Image.open(cect_path).convert("RGB"))
        return {"ncct_img": ncct_img, "cect_img": cect_img}

if __name__ == "__main__":
    import sys
    sys.path.append("/home/jeonghokim/CECT/src/cyclegan")
    from main import build_args
    from util.utils import Logger
    args = build_args()
    print(args)
    logger = Logger(0)
    logger.open("/home/jeonghokim/deprecated/asd.txt")
    train_dataset = UnAlignedDataset(args, logger, is_train = True)
    valid_dataset = UnAlignedDataset(args, logger, is_train=False)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4)
    valid_loader = DataLoader(valid_dataset, batch_size=4)
    for data in train_loader:
        ncct_img = data["ncct_img"]
        cect_img = data["cect_img"]
        import matplotlib.pyplot as plt
        plt.imshow(ncct_img[0].permute(1,2,0))
        plt.show()
        plt.imshow(cect_img[0].permute(1,2,0))
        plt.show()
        break






