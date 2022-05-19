import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .celebA import CelebA

def get_dataloader(args):
    if args.data_type == "cifar10":
        transform = T.Compose([
            T.Resize(size=(args.img_size, args.img_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=args.data_root_dir, train=True, transform=transform, download=True)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers,
        pin_memory=True, sampler=train_sampler)
    if args.data_type == "celebA":
        transform = T.Compose([
            T.Resize((args.img_size, args.img_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = CelebA(args, transform=transform)
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, pin_memory=True, sampler=train_sampler)
    return train_loader, train_sampler

        