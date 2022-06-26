import random

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T

from .custom_dataset import AFHQDataset, AFHQRefDataset, SingleDataset

def get_dataloader(args):
    crop = T.RandomResizedCrop(args.img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    random_crop = T.Lambda(lambda x: crop(x) if random.random() < 0.5 else x)
    transform = T.Compose([
        random_crop,
        T.Resize((args.img_size, args.img_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(0.5, 0.5)
    ])
    src_dataset = AFHQDataset(args, transform=transform)
    ref_dataset = AFHQRefDataset(args, transform=transform)
    if args.use_DDP:
        src_sampler = DistributedSampler(src_dataset, shuffle=True)
        ref_sampler = DistributedSampler(ref_dataset, shuffle=True)
        shuffle = False
    else:
        src_sampler = ref_sampler = None
        shuffle = True
    src_loader = DataLoader(src_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.n_workers, pin_memory=True, sampler=src_sampler, drop_last=True)
    ref_loader = DataLoader(ref_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.n_workers, pin_memory=True, sampler=ref_sampler, drop_last=True)
    return src_loader, ref_loader
def get_single_dataloader(data_dir, img_size, batch_size, imagenet_normalize=True, drop_last=False):
    if imagenet_normalize:
        H = 299
        W = 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        H = img_size
        W = img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.Resize((H,W)),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    dataset = SingleDataset(data_dir=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last)
    return loader