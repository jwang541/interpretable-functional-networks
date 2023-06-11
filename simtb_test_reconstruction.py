import argparse
import nibabel as nib
import scipy.stats
import torch

from models import Model
from utils import *



# Calculate lstsq loss of model on simtb data

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

    # flatten mask, model in, model out
    mask = torch.reshape(mask, (-1,))
    data = torch.reshape(data, (data.shape[0], -1))
    fns = torch.reshape(fns, (fns.shape[0], -1))

    data_masked = data[:, mask]
    fns_masked = fns[:, mask]

    # evaluate lstsq loss
    print('- Data reconstruction loss -')
    print(lstsq_loss(fns_masked.t(), data_masked.t()).item())
    print()
