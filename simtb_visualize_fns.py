import argparse
import math

import matplotlib.pyplot as plt
import torch
import nibabel as nib

from datasets import SimtbDataset
from utils import *
from models import Model



# Compute and visualize functional networks of simtb fMRI data (.nii)

# Usage: python simtb_visualize_fns.py -k NUMBER_FNS -w WEIGHTS -d DATA [-m MASK]

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###################################################################################################################

    with torch.no_grad():
        # load model weights
        model = Model(k_maps=args.k)
        model.load_state_dict(torch.load(args.weights))
        model = model.to(device)
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
        fns = model(data * mask, mask) * mask

    ###################################################################################################################

    # visualize learned functional networks
    fns = fns.cpu()
    fig, axes = plt.subplots(math.ceil(math.sqrt(args.k)), math.ceil(math.sqrt(args.k)), figsize=(10, 8))
    axes = axes.flatten()
    for i in range(args.k):
        axes[i].imshow(fns[i, 0], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title('FN {}'.format(i), fontsize=10, pad=2)
    plt.tight_layout()
    plt.show()
