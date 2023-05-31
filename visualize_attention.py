import argparse
import random

import matplotlib.pyplot as plt
import torch
import pandas as pd

from datasets import NiiDataset
from models import Model



# Usage: python visualize_attention.py -w WEIGHTS -d DATASET [-s SUBJECT]

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

        # visualize fmri datasets
        testset = NiiDataset(args.dataset, train=False, print_params=False)
        mri, mask = testset.__getitem__(args.subject)
        mri = torch.unsqueeze(mri, dim=0).float().to(device)
        mask = torch.unsqueeze(mask, dim=0).float().to(device)

        model_in = torch.unsqueeze(mri[0], dim=1) * mask[0]
        model_out = model(model_in) * mask[0]

        model_in = model_in.cpu()
        model_out = model_out.cpu()

        raw_attention = model.attention.last
        col_max, _ = torch.max(raw_attention, dim=0)
        normalized_attention = raw_attention / col_max

        frame = pd.DataFrame(normalized_attention.cpu().numpy()).astype('float')
        
        pd.options.display.float_format = '{:,.3f}'.format
        pd.set_option('display.max_rows', None)
        print(frame)

        # display 20 randomly selected time points
        tps = random.sample(range(model_in.shape[0]), min(30, model_in.shape[0]))
        tps.sort()

        fig, axes = plt.subplots(5, 6, figsize=(8, 8))
        axes = axes.flatten()
        for i in range(len(tps)):
            axes[i].imshow(model_in[tps[i], 0, 0], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title('t={}'.format(tps[i]), fontsize=10, pad=2)
        plt.tight_layout()
        plt.show()

