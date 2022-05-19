from os.path import join as opj

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .imagenet import ImageNet1K

def get_dataloader(args):
    if args.data_type == "Imagenet":
        train_transform = create_transform(
            input_size=args.img_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            interpolation='bicubic'
        )
        valid_transform = T.Compose([
            T.Resize((256,256)),
            T.CenterCrop(args.img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        train_dataset = ImageNet1K(opj(args.data_root_dir, args.data_type, "train"), transform=train_transform)
        valid_dataset = ImageNet1K(opj(args.data_root_dir, args.data_type, "val"), transform=valid_transform)

        if args.use_DDP: train_sampler = DistributedSampler(train_dataset, shuffle=True)
        else: train_sampler = None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, pin_memory=False, sampler=train_sampler, shuffle=train_sampler is None)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.n_workers, pin_memory=False)

        return train_loader, valid_loader, train_sampler
    
