import argparse
import numpy as np
import scipy
import scipy.stats
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt

from datasets import NiiDataset
from utils import *
from models import Model



# Evaluate the model on a .nii formatted simtb dataset

# Usage: python test_attention.py -w WEIGHTS -d DATASET [-s SUBJECT]

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, help='model weights (.pt file)', required=True)
    parser.add_argument('-d', '--dataset', type=str, help='simtb .nii dataset', required=True)
    parser.add_argument('-s', '--subject', type=int, help='subject index', default=0)

    # parse and print command line arguments
    args = parser.parse_args()
    print('weights path:', args.weights)
    print('dataset path:', args.dataset)
    print('subject #:', args.subject)
    print()

    ###################################################################################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        model = Model(k_maps=20, debug=True)
        model.load_state_dict(torch.load(args.weights))
        model = model.to(device)
        model.eval()

        # testset = NiiDataset(args.dataset, train=False, print_params=False, normalization='voxelwise')
        testset = NiiDataset(args.dataset, train=False, print_params=True, normalization='voxelwise')
        
        mri, mask = testset.__getitem__(args.subject)
        mri =  mri.float().to(device)
        mask = mask.bool().to(device)

        model_in = mri * mask
        model_in, rician_indices = add_rician_noise(model_in, mask, 0.25, std=1.0)
        model_in, affine_indices = add_affine2d_noise(model_in, mask, 0.25, max_trans=0.05, max_angle=5.0)

        model_out = model(model_in, mask)

        mask_flat = torch.reshape(mask, (-1,))
        x = torch.reshape(model_in, (model_in.shape[0], -1))
        y = torch.reshape(model_out, (model_out.shape[0], -1))

        x = x[:, mask_flat].cpu().numpy()
        y = y[:, mask_flat].cpu().numpy()

        ###############################################################################################################

        # Print correlation between each time point and each FN
        
        correlations = np.zeros((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                r, _ = scipy.stats.pearsonr(x[i], y[j])
                correlations[i, j] = r

        frame = pd.DataFrame(correlations).astype('float')
        pd.options.display.float_format = '{:,.3f}'.format
        pd.set_option('display.max_rows', None)
        print('- Correlations -')
        print(frame)
        print()

        ###############################################################################################################

        # Print r^2 values between each time point and FNs

        explained_variance = np.square(correlations)

        frame = pd.DataFrame(explained_variance).astype('float')
        pd.options.display.float_format = '{:,.5f}'.format
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
        model_out = model_out.cpu()

        raw_attention = model.attention.last
        col_max, _ = torch.max(raw_attention, dim=0)
        normalized_attention = raw_attention / col_max

        frame = pd.DataFrame(normalized_attention.cpu().numpy()).astype('float')
        
        pd.options.display.float_format = '{:,.3f}'.format
        pd.set_option('display.max_rows', None)

        print('- Attention -')
        print(frame)
        print()

        ###############################################################################################################

        # Display 20 randomly selected time points

        tps = random.sample(range(model_in.shape[0]), min(30, model_in.shape[0]))
        tps.sort()

        fig, axes = plt.subplots(5, 6, figsize=(8, 8))
        axes = axes.flatten()
        for i in range(len(tps)):
            axes[i].imshow(model_in[tps[i], 0], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title('t={}'.format(tps[i]), fontsize=10, pad=2)
        plt.tight_layout()
        plt.show()



