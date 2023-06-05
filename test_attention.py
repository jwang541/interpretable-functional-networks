import argparse
import sys
import numpy as np
import scipy
import scipy.stats
import torch
import pandas as pd

from datasets import NiiDataset
from models import Model



# Evaluate the model on a .nii formatted simtb dataset

# Usage: test_simtb.py -w WEIGHTS

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
        model = Model(k_maps=20)
        model.load_state_dict(torch.load(args.weights))
        model = model.to(device)
        model.eval()

        # visualize fmri datasets
        testset = NiiDataset(args.dataset, train=False, print_params=False, normalization='voxelwise')
        mri, mask = testset.__getitem__(args.subject)
        mri = torch.unsqueeze(mri, dim=0).float().to(device)
        mask = torch.unsqueeze(mask, dim=0).bool().to(device)

        model_out = model(mri[0], mask[0])

        mask = torch.reshape(mask, (-1,))
        x = torch.reshape(mri[0], (mri[0].shape[0], -1))
        y = torch.reshape(model_out, (model_out.shape[0], -1))

        x = x[:, mask].cpu().numpy()
        y = y[:, mask].cpu().numpy()

        ###############################################################################################################

        # Test 1: Compute correlation between each time point and each FN
        
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

        explained_variance = np.square(correlations)
        explained_variance = explained_variance / explained_variance.max(axis=0)

        frame = pd.DataFrame(explained_variance).astype('float')
        pd.options.display.float_format = '{:,.5f}'.format
        pd.set_option('display.max_rows', None)
        print('- Scaled r^2 -')
        print(frame)


