import os
import argparse

import numpy as np
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

    # make output directory if it doesn't already exist
    if not os.path.exists('./out'):
        os.makedirs('./out')

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
        # add noise to input
        model_in = data * mask
        model_in, rician_indices = add_rician_noise(model_in, mask, 0.25, std=0.25)
        model_in, affine_indices = add_affine2d_noise(model_in, mask, 0.25, max_trans=0.05, max_angle=5.0)

        # estimate functional networks
        fns = model(model_in, mask) * mask

    ###################################################################################################################
    
    # flatten mask, model in, model out
    mask = torch.reshape(mask, (-1,))

    x = torch.reshape(model_in, (model_in.shape[0], -1))
    y = torch.reshape(fns, (fns.shape[0], -1))

    x = x[:, mask].cpu().numpy()
    y = y[:, mask].cpu().numpy()


    ###################################################################################################################

    # Print correlation between each time point and each FN
        
    correlations = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            r, _ = scipy.stats.pearsonr(x[i], y[j])
            correlations[i, j] = r

    frame = pd.DataFrame(correlations).astype('float')
    pd.options.display.float_format = '{:,.3f}'.format
    pd.set_option('display.max_rows', None)
    print('- Time point & FN r -')
    print(frame)
    print()

    ###############################################################################################################

    # Print r^2 values between each time point and FNs

    explained_variance = np.square(correlations)

    frame = pd.DataFrame(explained_variance).astype('float')
    pd.options.display.float_format = '{:,.3f}'.format
    pd.set_option('display.max_rows', None)
    print('- Time point & FN r^2 -')
    print(frame)
    print()

    ###############################################################################################################

    # Print which time points had noise added
    print('- Noise indices -')
    print('Rician:', rician_indices)
    print('Affine:', affine_indices)
    print()

    ###############################################################################################################

    # Print attention values

    model_in = model_in.cpu()
    model_out = fns.cpu()

    raw_attention = model.attention.last
    col_max, _ = torch.max(raw_attention, dim=0)
    normalized_attention = raw_attention / col_max

    frame = pd.DataFrame(normalized_attention.cpu().numpy()).astype('float')
    
    pd.options.display.float_format = '{:,.3f}'.format
    pd.set_option('display.max_rows', None)

    print('- Attention -')
    print(frame)
    print()