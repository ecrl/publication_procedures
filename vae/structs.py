from typing import List

from torch.utils.data import Dataset
import torch
import numpy


class SMILESDataset(Dataset):

    def __init__(self, smiles_vectors: 'numpy.array'):

        self.shape = smiles_vectors.shape
        self.X = torch.as_tensor(smiles_vectors).type(torch.float32)
        # self.X = torch.flatten(
        #     torch.as_tensor(smiles_vectors), start_dim=1
        # ).type(torch.float32)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx: int) -> 'torch.tensor':

        return self.X[idx]

    @property
    def shape_flat(self):

        return self.X.shape
