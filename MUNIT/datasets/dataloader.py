from torch.utils.data import DataLoader
from .custom_dataset import UnAlignedDataset, TestDataset
def get_dataloader(args):
    train_dataset = UnAlignedDataset(args)
    test_dataset = TestDataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True)
    return train_loader, test_loader
    

    