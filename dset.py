import numpy as np
from torch.utils.data import Dataset


class DSet(Dataset):
    def __init__(self, data, label):
        self.x = data
        self.y = label

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def size(self):
        return self.__len__()
