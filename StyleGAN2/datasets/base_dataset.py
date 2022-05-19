from abc import abstractmethod
from abc import abstractmethod
from torch.utils.data import Dataset, DataLoader

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def __name__(self):
        raise NotImplementedError
    @abstractmethod
    def __check__(self):
        raise NotImplementedError
    
        