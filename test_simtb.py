import argparse
import numpy as np
import scipy
import scipy.stats
import torch

from datasets import NiiDataset
from models import Model
from utils import lstsq_loss



# Evaluate the model on a .nii formatted simtb dataset

# Usage: test_simtb.py -w WEIGHTS

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, help='model weights (.pt file)', required=True)

    # parse and print command line arguments
    args = parser.parse_args()
    print('weights path:', args.weights)
    print()

    ###################################################################################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        model = Model(k_maps=20)
        model.load_state_dict(torch.load(args.weights))
        model = model.to(device)
        model.eval()

        testset = NiiDataset('./data/simtb_data', train=False, print_params=False)
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

        ###############################################################################################################

        # Test 1: evaluate data reconstruction loss on testing dataset
        
        print('- Data reconstruction loss -')
        model_outputs = []
        eval_loss = 0
        for i, data in enumerate(testloader, 0):
            mri, mask = data
            mri = mri.float().to(device)
            mask = mask.bool().to(device)

            for n in range(mri.shape[0]):
                model_in = torch.unsqueeze(mri[n], dim=1) * mask[n]
                model_out = model(model_in) * mask[n]

                mask_flat = torch.flatten(mask[n])
                in_flat = torch.reshape(model_in, (model_in.shape[0], -1))
                out_flat = torch.reshape(model_out, (model_out.shape[0], -1))

                in_masked = in_flat[:, mask_flat]
                out_masked = out_flat[:, mask_flat]
                model_outputs.append(out_masked.cpu())

                loss = lstsq_loss(out_masked.t(), in_masked.t())
                eval_loss += loss.item()

                print('Subject {}: {}'.format(i, loss.item()))

        print('Average: ', eval_loss / len(testset))
        print()

        ###############################################################################################################

        # Test 2: compare to simtb ground truth FNs (all testset subjects)

        print('- Ground truth spatial correlation -')
        correlations = np.zeros(shape=(20, 20))

        # calculate correlations between learned FNs and ground truth FNs,
        # summed over all test subject data
        for i in range(len(testset)):
            # get mask
            _, mask = testset.__getitem__(i)
            mask_mat = torch.reshape(mask, (-1,))

            # load ground truth FNs
            sim_filename = './data/simtb_data/data/sim_subject_{0:0=3d}_SIM.mat'.format(i + 81)
            sim_data = scipy.io.loadmat(sim_filename)
            mask_mat_np = mask_mat.cpu().numpy()
            gt_fns_mat_masked = np.stack([
                np.extract(mask_mat_np, sim_data['SM'][k])
                for k in range(sim_data['SM'].shape[0])
            ])

            # update correlation matrix between ground truth and { base and se outputs }
            for a in range(gt_fns_mat_masked.shape[0]):
                for b in range(model_outputs[i].shape[0]):
                    r, _ = scipy.stats.pearsonr(gt_fns_mat_masked[a], model_outputs[i][b])
                    correlations[a, b] += r

        # perform linear sum assignment over summed correlations to get group atlas
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-1.0 * correlations)

        # calculate and output correlations between learned FNs and group atlas
        for i in range(20):
            # load ground truth FNs
            sim_filename = './data/simtb_data/data/sim_subject_{0:0=3d}_SIM.mat'.format(i + 81)
            sim_data = scipy.io.loadmat(sim_filename)
            mask_mat_np = mask_mat.cpu().numpy()
            gt_fns_mat_masked = np.stack([
                np.extract(mask_mat_np, sim_data['SM'][k])
                for k in range(sim_data['SM'].shape[0])
            ])

            # get outputs corresponding to the current subject
            model_output = model_outputs[i]

            print("Subject {}".format(i + 81))
            sum_correlations = 0
            for j in range(20):
                # calculate correlation between output and current ground truth FN
                correlation, _ = scipy.stats.pearsonr(gt_fns_mat_masked[j], model_output[col_ind[j]])
                sum_correlations += correlation

                # print correlations and average correlation
                print('FN {}\t\tMODEL {} {:.5f}'.format(
                    j,
                    col_ind[j],
                    correlation,
                ))
            print('Average\t\tBASE {:.5f}'.format(sum_correlations / 20.0))
            print()

        ###############################################################################################################

        # Test 3: analyze attention scores
