from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class BaseDataset(Dataset, ABC):
    def __init__(self, args):
        self.args = args
        self.A_paths = None
        self.B_paths = None
    @abstractmethod
    def name(self):
        return "BaseDataset"
        
