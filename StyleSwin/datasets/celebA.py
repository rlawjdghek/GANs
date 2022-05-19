from glob import glob
from os.path import join as opj

from PIL import Image
from torch.utils.data import Dataset

class CelebA(Dataset):
    def __init__(self, args, transform=None):
        self.img_paths = glob(opj(args.data_root_dir, "CelebA_HQ", "*"))
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path)
        img = self.transform(img)
        return img

