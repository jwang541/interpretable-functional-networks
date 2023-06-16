import os
import os.path
import random

import nibabel as nib
import scipy
import torch
from torch.utils.data import Dataset



class NoisyPairedDataset(Dataset):
    def __init__(self, dataset, noisy_dataset, p, normalization, eps=1e-9):
        self.dataset = dataset
        self.noisy_dataset = dataset

        if len(dataset) != len(noisy_dataset):
            raise Exception('unequal dataset lengths')
        
        self.p = p
        self.normalization = normalization
        self.eps = eps

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        with torch.no_grad():
            data, mask = self.dataset.__getitem__(idx)
            noisy_data, _ = self.noisy_dataset.__getitem__(idx)

            mixed_data = data.clone()
            mixed_indices = []

            for t in range(data.shape[0]):
                if random.random() < self.p:
                    mixed_data[t] = noisy_data[t]
                    mixed_indices.append(t)

            if self.normalization == 'global':
                std1, mu1 = torch.std_mean(data)
                data = (data - mu1) / (std1 + self.eps) * mask
                std2, mu2 = torch.std_mean(mixed_data)
                mixed_data = (mixed_data - mu2) / (std2 + self.eps) * mask
            elif self.normalization == 'voxelwise':
                std1, mu1 = torch.std_mean(data, dim=0) 
                data = (data - mu1) / (std1 + self.eps) * mask
                std2, mu2 = torch.std_mean(mixed_data, dim=0)
                mixed_data = (mixed_data - mu2) / (std2 + self.eps) * mask
            elif self.normalization == 'none':
                data = data * mask
                mixed_data = mixed_data * mask
            else:
                raise Exception('unknown normalization type')
            
            # print(data.shape)
            # print(mixed_data.shape)
            # print(mask.shape)

            # print(print(torch.isnan(data).any()))
            # print(print(torch.isnan(mixed_data).any()))

            return data, mixed_data, mask, mixed_indices

