import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .celebA import CelebA

def get_dataloader(args):
    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = CelebA(args, transform=transform)
    if args.use_DDP: train_sampler = DistributedSampler(train_dataset)
    else: train_sampler = None
    
    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size*2), num_workers=args.n_workers, pin_memory=True, sampler=train_sampler, shuffle=train_sampler is None)
    return train_loader, train_sampler

        