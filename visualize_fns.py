import argparse

import matplotlib.pyplot as plt
import torch

from datasets import SimtbDataset
from utils import *
from models import Model



# Usage: python visualize_fns.py -w WEIGHTS -d DATASET [-s SUBJECT]

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
        testset = SimtbDataset(args.dataset, train=False, print_params=True, normalization='voxelwise')
        mri, mask = testset.__getitem__(args.subject)
        mri = mri.float().to(device)
        mask = mask.float().to(device)

        model_in = mri * mask

        model_out = model(model_in, mask)
        model_out = model_out.cpu()

        # visualize learned FNs of a single subject
        fig, axes = plt.subplots(4, 5, figsize=(10, 8))
        axes = axes.flatten()
        for i in range(20):
            axes[i].imshow(model_out[i, 0], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title('FN {}'.format(i), fontsize=10, pad=2)
        plt.tight_layout()
        plt.show()

