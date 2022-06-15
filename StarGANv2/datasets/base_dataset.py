from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class BaseDataset(Dataset, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args