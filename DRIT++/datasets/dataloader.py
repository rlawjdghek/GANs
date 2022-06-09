from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .custom_dataset import UnAlignedDataset, TestDataset_single, TestDataset_pair

def get_dataloader(args,):
    train_dataset = UnAlignedDataset(args, is_train=True)
    valid_dataset = UnAlignedDataset(args, is_train=False)
    
    sampler = None
    shuffle = True
    if args.use_DDP:
        sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.n_workers, pin_memory=True, sampler=sampler, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.n_workers,
    pin_memory=True, drop_last=True)
    return train_loader, valid_loader
def get_test_dataloader(args, test_data_dir, is_single=True):
    if is_single:
        test_dataset = TestDataset_single(args, test_data_dir)
    else:
        test_dataset = TestDataset_pair(args, test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.n_workers, pin_memory=True)
    return test_loader
    
