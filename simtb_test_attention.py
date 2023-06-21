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



# Print relevant values for analyzing attention

# Usage: simtb_test_attention.py -k NUMBER_FNS -w WEIGHTS -d DATA [-m MASK]

# Required:
#   -k            : number of functional networks (must match model weights)
#   -w, --weights : model weights file
#   -d, --data    : .nii fMRI data file

# Optional:
#   -m, --mask    : .nii fMRI mask file, if not provided then no mask will be used

if __name__ == '__main__':
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, help='number of functional networks (must match model weights)', required=True)
    parser.add_argument('-w', '--weights', type=str, help='path to weights file', required=True)
    parser.add_argument('-d', '--data', type=str, help='path to .nii fMRI data file', required=True)
    parser.add_argument('-m', '--mask', type=str, help='path to .nii fMRI mask file', required=False)

    # print arguments
    args = parser.parse_args()
    print('- Arguments -')
    print('Number of functional networks (k):', args.k)
    print('Weights file:', args.weights)
    print('Data file:', args.data)
    print('Mask file:', args.mask if args.mask is not None else 'N/A')
    print()

    ###################################################################################################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###################################################################################################################

    with torch.no_grad():
        # load model weights
        model = Model(k_maps=args.k, debug=True)
        model = model.to(device)
        model.load_state_dict(torch.load(args.weights))
        model.eval()

    ###################################################################################################################

    with torch.no_grad():
        # load and preprocess fMRI data
        data_nii = nib.load(args.data)
        data = data_nii.get_fdata()
        data = torch.from_numpy(data)
        data = data.to(device)
        data = torch.permute(data, (3, 2, 1, 0))

        # voxelwise normalization
        std, mu = torch.std_mean(data, dim=0)
        data = (data - mu) / std
        data = data.float()

        # load and preprocess fMRI mask
        if args.mask is None:
            mask = torch.ones_like(data[0], dtype=bool)
        else:
            mask_nii = nib.load(args.mask)
            mask = mask_nii.get_fdata()
            mask = torch.from_numpy(mask)
            mask = mask.to(device)
            mask = torch.permute(mask, (2, 1, 0))
            mask = torch.greater(mask, 0.01)

    ###################################################################################################################
        
    with torch.no_grad():
        # estimate functional networks
        model_in = data * mask
        fns = model(model_in, mask) * mask

    ###################################################################################################################
    
    # flatten mask, model in, model out
    mask = torch.reshape(mask, (-1,))

    x = torch.reshape(model_in, (model_in.shape[0], -1))
    y = torch.reshape(fns, (fns.shape[0], -1))

    x = x[:, mask].cpu().numpy()
    y = y[:, mask].cpu().numpy()

    ###################################################################################################################

    # Print time courses

    tcs, _, _, _ = linalg.lstsq(y.T, x.T, rcond=None)
    tcs = tcs.T
    print(tcs.shape)

    frame = pd.DataFrame(tcs).astype('float')
    pd.options.display.float_format = '{:,.3f}'.format
    pd.set_option('display.max_rows', None)
    print('- Time courses -')
    print(frame)
    print()

    ###################################################################################################################

    # Print attention values

    model_in = model_in.cpu()
    model_out = fns.cpu()

    raw_attention = model.last_attention
    col_max, _ = torch.max(raw_attention, dim=0)
    normalized_attention = raw_attention / col_max

    frame = pd.DataFrame(normalized_attention.cpu().numpy()).astype('float')
    
    pd.options.display.float_format = '{:,.3f}'.format
    pd.set_option('display.max_rows', None)

    print('- Attention -')
    print(frame)
    print()