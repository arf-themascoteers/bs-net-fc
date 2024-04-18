import numpy as np
from torch.utils.data import Dataset
import torch


class DSet(Dataset):
    def __init__(self, data, label):
        self.x = torch.tensor(data, dtype=torch.float32)
        self.y = torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def size(self):
        return self.__len__()
