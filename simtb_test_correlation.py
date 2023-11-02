import os
import argparse
import scipy
import scipy.io as sio

import numpy as np
import torch
import nibabel as nib

from models import Model
from utils import *



# Estimate functional networks, save the results to .nii files in ./out

# Usage: deploy.py -k NUMBER_FNS -w WEIGHTS -d DATA [-m MASK] -s SOURCE

# Required:
#   -k            : number of functional networks (must match model weights)
#   -w, --weights : model weights file
#   -d, --data    : .nii fMRI data file
#   -s, --source : simtb source maps (.mat file)

# Optional:
#   -m, --mask    : .nii fMRI mask file, if not provided then no mask will be used

if __name__ == '__main__':
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, help='number of functional networks (must match model weights)', required=True)
    parser.add_argument('-w', '--weights', type=str, help='path to weights file', required=True)
    parser.add_argument('-d', '--data', type=str, help='path to .nii fMRI data file', required=True)
    parser.add_argument('-m', '--mask', type=str, help='path to .nii fMRI mask file', required=False)
    parser.add_argument('-s', '--source', type=str, help='path to .mat simtb source file', required=True)

    # print arguments
    args = parser.parse_args()
    print('- Arguments -')
    print('Number of functional networks (k):', args.k)
    print('Weights file:', args.weights)
    print('Data file:', args.data)
    print('Mask file:', args.mask if args.mask is not None else 'N/A')
    print('Source file:', args.source)
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
        model = Model(k_maps=args.k)
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

        # voxelwise normalization
        # std, mu = torch.std_mean(data, dim=0)

        # global normalization
        std, mu = torch.std_mean(data)

        data = (data - mu) / (std + 1e-8)
        data = data.float()

    ###################################################################################################################
    
    with torch.no_grad():
        # estimate functional networks
        fns = model(data * mask, mask) * mask

    ###################################################################################################################

    # load simtb source maps 
    data = sio.loadmat(args.source)
    source_maps = data['SM']
    time_courses = data['TC']

    ###################################################################################################################

    fns = torch.reshape(fns, (fns.shape[0], -1))
    fns = fns.cpu()
    fns = fns.numpy()

    mask = torch.reshape(mask, (-1,))
    mask = mask.cpu()
    mask = mask.numpy()

    fns_masked = fns[:, mask]
    sms_masked = source_maps[:, mask]

    # calculate pearson r between each functional network and source map
    correlations = np.zeros(shape=(args.k, args.k))
    for i in range(args.k):
        for j in range(args.k):
            r, _ = scipy.stats.pearsonr(sms_masked[i], fns_masked[j])
            correlations[i, j] = r

    # perform linear sum assignment over correlations to match FNs to SMs
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-1.0 * correlations)

    # print correlations
    r_sum = 0
    print('- Ground truth spatial correlation -')
    for i in range(args.k):
        correlation, _ = scipy.stats.pearsonr(sms_masked[i], fns_masked[col_ind[i]])
        print('SM {}\tFN {}\t{:.5f}'.format(i, col_ind[i], correlation))
        r_sum += correlation
    print('Average\t\t{:.5f}'.format(r_sum / args.k))

   