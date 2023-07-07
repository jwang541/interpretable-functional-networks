import os
import os.path
import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset



class HcpDataset(Dataset):
    def __init__(self, data_index, mask_file, print_params=False, normalization='global', 
                 crop_s=None, crop_e=None,
                 eps=1e-8):
        self.data_files = []
        with open(data_index, 'r') as f:
            for line in f:
                if len(line.strip()) != 0:
                    self.data_files.append(line.strip())

        self.mask_file = mask_file

        self.normalization = normalization

        self.crop_s = crop_s
        self.crop_e = crop_e

        self.eps = eps

        if print_params:
            print('path:', data_index)
            print('# subjects:', len(self.data_files))
            print('normalization: ', self.normalization)

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, i):
        with torch.no_grad():
            data_proxy = nib.load(self.data_files[i])
            crop_s = (0, 0, 0) if self.crop_s is None else self.crop_s
            crop_e = (data_proxy.shape[0:3]) if self.crop_e is None else self.crop_e
            tps = random.sample(range(data_proxy.shape[3]), 20)
            tps.sort()

            data = data_proxy.dataobj[crop_s[0]:crop_e[0],
                                      crop_s[1]:crop_e[1],
                                      crop_s[2]:crop_e[2], 
                                      :]
            data = data[..., tps]
            data = data.astype(np.float32)
            data = torch.from_numpy(data)
            data = torch.permute(data, (3, 2, 1, 0))
            
            mask_proxy = nib.load(self.mask_file)
            mask = mask_proxy.get_fdata(dtype=np.float32)
            mask = mask[crop_s[0]:crop_e[0],
                        crop_s[1]:crop_e[1],
                        crop_s[2]:crop_e[2]]
            mask = torch.from_numpy(mask)
            mask = torch.permute(mask, (2, 1, 0))
            mask = mask.bool()

            if self.normalization == 'global':
                std, mu = torch.std_mean(data[:, mask])
                data = (data - mu) / (std + self.eps) * mask
            elif self.normalization == 'voxelwise':
                std, mu = torch.std_mean(data, dim=0) 
                data = (data - mu) / (std + self.eps) * mask
            else:
                raise Exception('unknown normalization type')

            return data, mask

