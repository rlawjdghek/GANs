from torchvision.datasets import LSUN
import torchvision.transforms as T
from torch.utils.data import DataLoader

transform = T.Compose([
    T.Resize((64,64)),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
train_bedroom_dataset = LSUN("/home/data/LSUN", classes=["bedroom_train"], transform=transform)
valid_bedroom_dataset = LSUN("/home/data/LSUN", classes=["bedroom_val"], transform=transform)
train_bedroom_loader = DataLoader(train_bedroom_dataset, batch_size=32, shuffle=True)
valid_bedroom_loader = DataLoader(valid_bedroom_dataset, batch_size=32, shuffle=False)
print(len(train_bedroom_dataset))
print(len(valid_bedroom_loader))

    

