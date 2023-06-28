import os
import argparse

import torch
import torch.nn as nn

from datasets import HcpDataset
from models import Model
from utils import *

import resource




if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()

    # train mode arguments
    train_mode_args = parser.add_mutually_exclusive_group(required=True)
    train_mode_args.add_argument('-p', '--pretrain', action='store_true', help='pretrain the model')
    train_mode_args.add_argument('-f', '--finetune', action='store_true', help='finetune the model')

    # data arguments
    # parser.add_argument('-d', '--dataset', type=str, required=True, help='path to dataset (must be compatible with SimtbDataset class)')
    parser.add_argument('-k', type=int, help='number of functional networks (must match checkpoint weights)', required=True)

    # hyperparameter arguments
    parser.add_argument('-c', '--checkpoint', type=str, help='path to checkpoint weights file')
    parser.add_argument('-e', '--epochs', type=int, default=300, help='number of epochs (default: 300)')
    parser.add_argument('-l', '--lr', type=float, default=0.001, help='learning rate (default: 0.0001)')
    parser.add_argument('-t', '--trade_off', type=float, default=10, help='hoyer trade off parameter (default: 10)')

    # parse and print command line arguments
    args = parser.parse_args()
    print('- Training parameters -')
    print('train mode:', 'finetune' if args.finetune else 'pretrain')
    print('checkpoint path:', args.checkpoint)
    print('epochs:', args.epochs)
    print('learning rate:', args.lr)
    if args.finetune:
        print('sparsity trade-off:', args.trade_off)
    print()

    ###################################################################################################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###################################################################################################################

    print('- Trainset parameters -')
    trainset = HcpDataset(data_index='/cbica/home/lihon/comp_space/bbl_pnc_resting/hcp_sm_data/hcp_sm6_t400_tra.txt',
                          mask_file='/cbica/home/lihon/comp_space/bbl_pnc_resting/rnn_autoencoder/scripts/mask_thr0p5_wmparc.2_cc_3mm.nii.gz',
                          print_params=True,
                          normalization='voxelwise')
    print()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

    ###################################################################################################################

    model = Model(k_maps=args.k)
    model = model.to(device)
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    ###################################################################################################################

    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    ###################################################################################################################

    if not os.path.exists('./out'):
        os.makedirs('./out')

    ###################################################################################################################

    for epoch in range(args.epochs):
        if epoch % 1 == 0:
            torch.save(model.state_dict(), './out/e{}.pt'.format(epoch))

        model.train()
        train_loss = 0
        for i, data in enumerate(trainloader):
            mri, mask = data
            mri = mri.float().to(device)
            mask = mask.bool().to(device)

            for n in range(mri.shape[0]):
                optimizer.zero_grad()

                model_in = mri[n]
                # model_in = model_in[0:10]
                model_out = model(model_in, mask[n])
                # print('forward', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 'GB')

                mask_flat = torch.flatten(mask[n])
                in_flat = torch.reshape(model_in, (model_in.shape[0], -1))
                out_flat = torch.reshape(model_out, (model_out.shape[0], -1))

                in_masked = in_flat[:, mask_flat]
                out_masked = out_flat[:, mask_flat]

                if args.finetune:
                    loss = lstsq_loss(out_masked.t(), in_masked.t()) + args.trade_off * hoyer_loss(out_masked)
                else:
                    loss = clustering_loss(out_masked, in_masked)
                
                loss.backward()
                # print('backward', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 'GB')

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                train_loss += loss.item()

                optimizer.step()

                # print('optimizer', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 'GB')
                # exit()

        print(epoch, train_loss)

    torch.save(model.state_dict(), './out/e{}.pt'.format(args.epochs))
