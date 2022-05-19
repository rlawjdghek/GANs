from torchvision.datasets import ImageFolder

class ImageNet1K(ImageFolder):
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None: 
            sample = self.transform(sample)
        return sample, target