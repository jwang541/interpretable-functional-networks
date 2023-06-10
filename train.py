import os
import argparse

import torch
import torch.nn as nn

from datasets import NiiDataset
from models import Model
from utils import *



# Usage: train.py [-h] (-p | -f) [-c CHECKPOINT] [-e EPOCHS] [-l LR] [-t TRADE_OFF]

# Required:
#   -p, --pretrain : pretraining (lstsq) loss function
#   -f, --finetune : finetuning (clustering) loss function

# Optional:
#   -c, --checkpoint CHECKPOINT : path to a checkpoint weights file (.pt)
#   -e, --epochs EPOCHS         : number of epochs (default: 300)
#   -l, --lr LR                 : learning rate (default: 0.0001)
#   -t, --trade-off TRADE_OFF   : sparsity trade-off value, only used for finetune (default: 10)

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()

    # train mode arguments
    train_mode_args = parser.add_mutually_exclusive_group(required=True)
    train_mode_args.add_argument('-p', '--pretrain', action='store_true', help='pretrain the model')
    train_mode_args.add_argument('-f', '--finetune', action='store_true', help='finetune the model')

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

    print('- Trainset parameters -')
    trainset = NiiDataset('./data/simtb_data', train=True, print_params=True, normalization='voxelwise')
    print()

    print('- Testset parameters -')
    testset = NiiDataset('./data/simtb_data', train=False, print_params=True, normalization='voxelwise')
    print()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    model = Model(k_maps=20)
    model = model.to(device)
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    if not os.path.exists('./out'):
        os.makedirs('./out')

    for epoch in range(args.epochs):
        if epoch % 10 == 0:
            torch.save(model.state_dict(), './out/e{}.pt'.format(epoch))

        model.train()
        train_loss = 0
        for i, data in enumerate(trainloader):
            mri, mask = data
            mri = mri.float().to(device)
            mask = mask.bool().to(device)

            for n in range(mri.shape[0]):
                optimizer.zero_grad()

                model_in = mri[n] * mask[n]
                model_in, rician_indices = add_rician_noise(model_in, mask[n], 0.25, std=1.0)
                model_in, affine_indices = add_affine2d_noise(model_in, mask[n], 0.25, max_trans=0.05, max_angle=5.0)
                # print('Rician:', rician_indices)
                # print('Affine:', affine_indices)

                # Compute FNs with added noise
                model_out = model(model_in, mask[n])

                mask_flat = torch.flatten(mask[n])
                in_flat = torch.reshape(mri[n], (mri[n].shape[0], -1))
                out_flat = torch.reshape(model_out, (model_out.shape[0], -1))

                in_masked = in_flat[:, mask_flat]
                out_masked = out_flat[:, mask_flat]

                # Compute loss function without added noise; encourages model to be robust to noise
                if args.finetune:
                    loss = lstsq_loss(out_masked.t(), in_masked.t()) + args.trade_off * hoyer_loss(out_masked)
                else:
                    loss = clustering_loss(out_masked, in_masked)
                
                loss.backward()
                
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                train_loss += loss.item()

                optimizer.step()

        model.eval()
        eval_loss = 0
        for i, data in enumerate(testloader):
            mri, mask = data
            mri = mri.float().to(device)
            mask = mask.bool().to(device)

            for n in range(mri.shape[0]):
                model_in = mri[n] * mask[n]
                model_in, rician_indices = add_rician_noise(model_in, mask[n], 0.25, std=1.0)
                model_in, affine_indices = add_affine2d_noise(model_in, mask[n], 0.25, max_trans=0.05, max_angle=5.0)
                # print('Rician:', rician_indices)
                # print('Affine:', affine_indices)

                # Compute FNs with added noise
                model_out = model(model_in, mask[n])

                mask_flat = torch.flatten(mask[n])
                in_flat = torch.reshape(mri[n], (mri[n].shape[0], -1))
                out_flat = torch.reshape(model_out, (model_out.shape[0], -1))

                in_masked = in_flat[:, mask_flat]
                out_masked = out_flat[:, mask_flat]

                # Compute loss function without added noise; encourages model to be robust to noise
                if args.finetune:
                    loss = lstsq_loss(out_masked.t(), in_masked.t()) + args.trade_off * hoyer_loss(out_masked)
                else:
                    loss = clustering_loss(out_masked, in_masked)

                eval_loss += loss.item()

        print('[{}]\t\ttrain loss: {:.3f}\t\teval loss: {:.3f}'
              .format(epoch + 1, train_loss / len(trainloader.dataset), eval_loss / len(testloader.dataset)))

    torch.save(model.state_dict(), './out/e{}.pt'.format(args.epochs))
