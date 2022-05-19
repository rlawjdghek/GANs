from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .selfie2anime import Selfie2Anime

def get_dataloader(args, logger):
    if args.data_name.lower() == "selfie2anime":
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.Resize((args.img_size + 30, args.img_size + 30)),
            T.RandomCrop(args.img_size),
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
        ])
        test_transform = T.Compose([
            T.Resize((args.img_size, args.img_size)),
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = Selfie2Anime(args, logger, is_train=True, transform=train_transform)
        test_dataset = Selfie2Anime(args, logger, is_train=False, transform=test_transform)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.n_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        return train_loader, test_loader

    