from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .custom_dataset import UnAlignedDataset, AlignedDataset

def get_dataloader(args, logger):
    train_dataset = AlignedDataset(args, logger, is_train=True)
    valid_dataset = AlignedDataset(args, logger, is_train=False)
    sampler=None
    shuffle=True
    if args.use_DDP:
        sampler = DistributedSampler(train_dataset)
        shuffle=False
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle,
                              num_workers=args.num_workers, pin_memory=True, sampler=sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    return train_loader, valid_loader