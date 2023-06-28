import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import nibabel as nib

from models import Model
from utils import *



# Estimate functional networks, save the results to .nii files in ./out

# Usage: deploy.py -k NUMBER_FNS -w WEIGHTS -d DATA [-m MASK]

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
    device = 'cpu'

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

        # voxelwise normalization
        std, mu = torch.std_mean(data, dim=0)
        data = (data - mu) / (std + 1e-8)
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
        fns = model(data * mask, mask) * mask

    ###################################################################################################################

    # save each functional network to a .nii file
    timestr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    for i in range(fns.shape[0]):
        fn = fns[i]
        fn = torch.permute(fn, (2, 1, 0))
        fn = fn.cpu()
        fn = fn.numpy()
        fn_nii = nib.Nifti1Image(fn, affine=None)
        nib.save(fn_nii, './out/{}_fn{}'.format(timestr, i))

   