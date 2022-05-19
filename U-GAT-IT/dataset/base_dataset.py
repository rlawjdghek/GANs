from torch.utils.data import Dataset
from abc import abstractmethod

class BaseDataset(Dataset):
    def __init__(self): 
        super().__init__()
    @abstractmethod
    def __check__(self): pass
