from typing import Union, List
from glob import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CelebA_dataset(Dataset):
    def __init__(self, root_dir : str, mode : str = "train", transform = None, attributes : List = None):
        self.transform = transform
        self.attributes = attributes  # 내가 쓸 attributes
        self.file_names = os.listdir(os.path.join(root_dir, "img_align_celeba/"))
        self.img_paths = sorted([root_dir+"img_align_celeba/"+name for name in self.file_names])
        self.img_paths = self.img_paths[:-2000] if mode=="train" else self.img_paths[-2000:-1]
        self.anno_path = os.path.join(root_dir, "Anno/list_attr_celeba.txt")
        self.annotations = self.get_all_label()
        print(f"# imgs : {len(self.img_paths)}")
        
        
    def get_all_label(self):
        anno_file = open(self.anno_path, "r")
        lines = [line.strip() for line in anno_file]
        n_files = int(lines[0])
        attributes = lines[1].split()
        my_annotation = {} 
        for line in lines[2:]:
            annos = []
            file_name, *all_anno = line.split()
            for attr in self.attributes:
                idx = attributes.index(attr)
                annos.append(int(all_anno[idx]=="1"))
            my_annotation[file_name] = annos
            
        return my_annotation
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        data = {}
        img_path = self.img_paths[idx % len(self.img_paths)]
        img_name = img_path.split("/")[-1]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        label = torch.FloatTensor(self.annotations[img_name])
        data["img"] = img
        data["label"] = label
        return data
    
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor()
    ])
    attributes = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
    train_dataset = CelebA_dataset(root_dir = "/jupyterdata/CelebA/", transform = transform, attributes=attributes)
    test_dataset = CelebA_dataset(root_dir = "/jupyterdata/CelebA/", transform = transform, attributes=attributes, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=4)
    test_loader = DataLoader(test_dataset, batch_size=4)

    for data in train_loader:
        img = data["img"]
        label = data["label"]
        print(img.shape)
        print(label)
        break

    for data in test_loader:
        img = data["img"]
        label = data["label"]
        print(img.shape)
        print(label)
        break    