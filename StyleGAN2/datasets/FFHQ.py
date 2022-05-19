from os.path import join as opj
from glob import glob
import random
from PIL import Image

from .base_dataset import BaseDataset

#### TODO: image ram에 다 올리고 시간 계산 해보기
class FFHQDataset(BaseDataset):
    def __init__(self, args, logger, transform):
        super().__init__()
        self.args = args
        self.logger = logger
        self.transform = transform
        self.img_paths = sorted(glob(opj(self.args.data_root_dir, "*.png")))
        self.__check__()
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        return img        
    def __len__(self):
        return len(self.img_paths)
    def __check__(self):
        self.logger.write(f"# imgs : {len(self.img_paths)}\n")
    def __name__(self):
        return "FFHQDataset"

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    class args:
        data_root_dir = "/home/data/FFHQ_1024"
    sample_dataset = FFHQDataset(args)
    print(len(sample_dataset))
    for img in sample_dataset:
        print(img.shape)
        break

    
