from glob import glob
from os.path import join as opj

from PIL import Image

from .base_dataset import BaseDataset

class Selfie2Anime(BaseDataset):
    def __init__(self, args, logger, is_train, transform):
        super().__init__()
        self.args = args
        self.logger = logger
        self.transform = transform
        self.cate = "train" if is_train else "test"
        self.img_paths_A = sorted(glob(opj(self.args.data_root_dir, args.data_name, f"{self.cate}A", "*")))
        self.img_paths_B = sorted(glob(opj(self.args.data_root_dir, args.data_name, f"{self.cate}B", "*")))
        self.n_A = len(self.img_paths_A)
        self.n_B = len(self.img_paths_B)
        self.__check__()
    def __check__(self):
        self.logger.write(f"Dataset : {self.__class__.__name__}, category : {self.cate}, imgs_A : {len(self.img_paths_A)}, imgs_B : {len(self.img_paths_B)}\n")
    def __getitem__(self, idx):
        img_path_A = self.img_paths_A[idx % self.n_A]
        img_path_B = self.img_paths_B[idx % self.n_B]
        img_A = Image.open(img_path_A).convert("RGB")
        img_A = self.transform(img_A)
        img_B = Image.open(img_path_B).convert("RGB")
        img_B = self.transform(img_B)
        return img_A, img_B       
    def __len__(self):
        return max(self.n_A, self.n_B)

