import os
import os.path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset



class HcpDataset(Dataset):
    def __init__(self, data_index, mask_file, print_params=False, normalization='global', eps=1e-8):
        self.data_files = []
        with open(data_index, 'r') as f:
            for line in f:
                if len(line.strip()) != 0:
                    self.data_files.append(line.strip())

        self.mask_file = mask_file

        self.eps = eps
        self.normalization = normalization

        if print_params:
            print('path:', data_index)
            print('# subjects:', len(self.data_files))
            print('normalization: ', self.normalization)

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, i):
        with torch.no_grad():
            data = nib.load(self.data_files[i])
            data = data.get_fdata(dtype=np.float32)
            data = torch.from_numpy(data)
            data = torch.permute(data, (3, 2, 1, 0))

            mask = nib.load(self.mask_file)
            mask = mask.get_fdata(dtype=np.float32)
            mask = torch.from_numpy(mask)
            mask = torch.permute(mask, (2, 1, 0))
            mask = mask.bool()

            # TODO: temporary
            data = data[::20]

            if self.normalization == 'global':
                std, mu = torch.std_mean(data[:, mask])
                data = (data - mu) / (std + self.eps) * mask
            elif self.normalization == 'voxelwise':
                std, mu = torch.std_mean(data, dim=0) 
                data = (data - mu) / (std + self.eps) * mask
            else:
                raise Exception('unknown normalization type')

            return data, mask

