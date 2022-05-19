from matplotlib import transforms
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .FFHQ import FFHQDataset

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
def get_dataloader(args, logger):
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5), inplace=True)
    ])
    train_dataset = FFHQDataset(args, logger, transform=train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=args.n_workers,
        pin_memory=True
    )
    train_loader = sample_data(train_loader)
    return train_loader
    
