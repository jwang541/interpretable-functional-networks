import os
from datetime import datetime

import torch
import torch.nn as nn
import nibabel as nib

from models import Model
from utils import *
from config import *



# Estimate functional networks, save the results to .nii files in ./out

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
    device = 'cpu'

    ###################################################################################################################

    with torch.no_grad():
        # load model weights
        model = Model(k_maps=config.k)
        model = model.to(device)
        model.load_state_dict(torch.load(config.weights))
        model.eval()

    ###################################################################################################################

    with torch.no_grad():
        # load and preprocess fMRI data
        data_nii = nib.load(config.data)
        data = data_nii.get_fdata()
        crop_s = (0, 0, 0) if config.crop_s is None else config.crop_s
        crop_e = (data.shape[0:3]) if config.crop_e is None else config.crop_e
        data = data[crop_s[0]:crop_e[0],
                    crop_s[1]:crop_e[1],
                    crop_s[2]:crop_e[2], 
                    :]

        data = torch.from_numpy(data)
        data = data.to(device)
        data = torch.permute(data, (3, 2, 1, 0))

        # voxelwise normalization
        std, mu = torch.std_mean(data, dim=0)
        data = (data - mu) / (std + 1e-8)
        data = data.float()

        # load and preprocess fMRI mask
        if config.mask is None:
            mask = torch.ones_like(data[0], dtype=bool)
        else:
            mask_nii = nib.load(config.mask)
            mask = mask_nii.get_fdata()
            mask = mask[crop_s[0]:crop_e[0],
                        crop_s[1]:crop_e[1],
                        crop_s[2]:crop_e[2]]

            mask = torch.from_numpy(mask)
            mask = mask.to(device)
            mask = torch.permute(mask, (2, 1, 0))
            mask = torch.greater(mask, 0.01)

    ###################################################################################################################
    
    with torch.no_grad():
        # estimate functional networks
        fns = model(data * mask, mask) * mask

    ###################################################################################################################

    # make output directory if it doesn't already exist
    timestr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    outdir = './out/{}'.format(timestr)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # save each functional network to a .nii file
    for i in range(fns.shape[0]):
        fn = fns[i]
        fn = torch.permute(fn, (2, 1, 0))
        fn = fn.cpu()
        fn = fn.numpy()
        fn_nii = nib.Nifti1Image(fn, affine=None)
        nib.save(fn_nii, os.path.join(outdir, 'fn{}'.format(i)))

   