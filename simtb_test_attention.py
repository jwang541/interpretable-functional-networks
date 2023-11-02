import os
import argparse

import numpy as np
import numpy.linalg as linalg
import pandas as pd
import scipy
import torch
import nibabel as nib

from models import Model
from utils import *
from config import *



# Print relevant values for analyzing attention

if __name__ == '__main__':
    
    config = deploy_config()

    # print arguments
    print('- Arguments -')
    print('Number of functional networks (k):', config.k)
    print('Weights file:', config.weights)
    print('Data file:', config.data)
    print('Mask file:', config.mask if config.mask is not None else 'N/A')
    print()

    ###################################################################################################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###################################################################################################################

    with torch.no_grad():
        # load model weights
        model = Model(k_maps=config.k, debug=True)
        model = model.to(device)
        model.load_state_dict(torch.load(config.weights))
        model.eval()

    ###################################################################################################################

    with torch.no_grad():
        # load and preprocess fMRI data
        data_nii = nib.load(config.data)
        data = data_nii.get_fdata()
        data = torch.from_numpy(data)
        data = data.to(device)
        data = torch.permute(data, (3, 2, 1, 0))

        # voxelwise normalization
        # std, mu = torch.std_mean(data, dim=0)
        std, mu = torch.std_mean(data)
        gdata = (data - mu) / (std + 1e-8)
        gdata = gdata.float()

        vdata = (data - torch.mean(data, dim=0)) / (torch.std(data, dim=0) + 1e-8)
        vdata = vdata.float()

        # load and preprocess fMRI mask
        if config.mask is None:
            mask = torch.ones_like(data[0], dtype=bool)
        else:
            mask_nii = nib.load(config.mask)
            mask = mask_nii.get_fdata()
            mask = torch.from_numpy(mask)
            mask = mask.to(device)
            mask = torch.permute(mask, (2, 1, 0))
            mask = torch.greater(mask, 0.01)

    ###################################################################################################################
        
    with torch.no_grad():
        # estimate functional networks
        gdata = gdata * mask
        vdata = vdata * mask
        fns = model(gdata, mask) * mask

    ###################################################################################################################
    
    # Print attention values
    raw_attention = model.last_attention
    col_max, _ = torch.max(raw_attention, dim=0)
    normalized_attention = raw_attention / col_max

    frame = pd.DataFrame(normalized_attention.cpu().numpy()).astype('float')
    
    pd.options.display.float_format = '{:,.3f}'.format
    pd.set_option('display.max_rows', None)

    print('- Attention -')
    print(frame)
    print()

    ###################################################################################################################

    # flatten mask, model in, model out
    mask = torch.reshape(mask, (-1,))

    x = torch.reshape(vdata, (vdata.shape[0], -1))
    y = torch.reshape(fns, (fns.shape[0], -1))

    x = x[:, mask].cpu().numpy()
    y = y[:, mask].cpu().numpy()

    ###################################################################################################################

    # Print time courses

    tcs, _, _, _ = np.linalg.lstsq(y.T, x.T, rcond=None)
    tcs = tcs.T
    print(tcs.shape)

    frame = pd.DataFrame(tcs).astype('float')
    pd.options.display.float_format = '{:,.3f}'.format
    pd.set_option('display.max_rows', None)
    print('- Time courses -')
    print(frame)
    print()
