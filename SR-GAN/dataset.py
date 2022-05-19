import torchvision
from torchvision import transforms
import torch
from glob import glob
import os
from PIL import Image

class dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, w,h):
        self.img_paths = glob(os.path.join(root_dir, "*.jpg"))
        self.lr_transforms = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
        self.hr_transforms = transforms.Compose([
            transforms.Resize((h//4, w//4)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        img_lr = self.lr_transforms(img)
        img_hr = self.hr_transforms(img)
        return img_lr, img_hr    