import sys

import matplotlib.pyplot as plt
import torch

from datasets import NiiDataset
from models import Model



# Usage: python visualize_fns.py <WEIGHTS FILENAME>.pt

if __name__ == '__main__':

    # get weights path
    if len(sys.argv) != 2:
        raise Exception('Usage: python visualize_fns.py <WEIGHTS FILENAME>.pt')
    weights_path = sys.argv[1]

    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Model(k_maps=20)
        model.load_state_dict(torch.load(weights_path))
        model = model.to(device)

        model.eval()

        # visualize fmri datasets
        testset = NiiDataset('./data/simtb_data', train=False, print_params=False)
        mri, mask = testset.__getitem__(13)
        mri = torch.unsqueeze(mri, dim=0).float().to(device)
        mask = torch.unsqueeze(mask, dim=0).float().to(device)

        model_in = torch.unsqueeze(mri[0], dim=1) * mask[0]
        model_out = model(model_in) * mask[0]

        model_out = model_out.cpu()

        # visualize learned FNs of a single subject
        fig, axes = plt.subplots(4, 5, figsize=(10, 8))
        axes = axes.flatten()
        for i in range(20):
            axes[i].imshow(model_out[i, 0], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(str(i + 1), fontsize=10, pad=2)
        plt.tight_layout()
        plt.show()

