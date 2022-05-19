from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .custom_dataset import VideoDataset, VideoTestDataset

def get_dataloader(args):
    train_dataset = VideoDataset(args, is_train=True)
    valid_dataset = VideoDataset(args, is_train=False)
    sampler = None
    if args.use_DDP:
        sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=sampler is None, num_workers=args.n_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.n_workers, pin_memory=True)
    return train_loader, valid_loader
def get_test_dataloader(args, test_data_name):
    valid_dataset = VideoTestDataset(args, "valid")
    test_dataset = VideoTestDataset(args, test_data_name) 
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.n_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.n_workers, pin_memory=True)
    return valid_loader, test_loader