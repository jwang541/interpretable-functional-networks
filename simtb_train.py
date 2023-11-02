import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn


from datasets import SimtbDataset
from models import Model
from utils import *
from config import *



# Usage: simtb_train.py [-h] (-p | -f) -d DATASET [-c CHECKPOINT] [-e EPOCHS] [-l LR] [-t TRADE_OFF]

# Required:
#   -p, --pretrain : pretraining (lstsq) loss function
#   -f, --finetune : finetuning (clustering) loss function
#   -d, --dataset  : path to dataset (must be compatible with SimtbDataset class)

# Optional:
#   -c, --checkpoint CHECKPOINT : path to a checkpoint weights file (.pt)
#   -e, --epochs EPOCHS         : number of epochs (default: 300)
#   -l, --lr LR                 : learning rate (default: 0.0001)
#   -t, --trade-off TRADE_OFF   : sparsity trade-off value, only used for finetune (default: 10)

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    train_mode_args = parser.add_mutually_exclusive_group(required=True)
    train_mode_args.add_argument('-p', '--pretrain', action='store_true', help='pretrain the model')
    train_mode_args.add_argument('-f', '--finetune', action='store_true', help='finetune the model')
    args = parser.parse_args()

    if args.pretrain:
        config = simtb_pretrain_config()
    else:
        config = simtb_finetune_config()

    # parse and print command line arguments
    args = parser.parse_args()
    print('- Training parameters -')
    print('train mode:', 'pretrain' if config.mode == 0 else 'finetune')
    if config.mode == 1:
        print('sparsity trade-off:', config.tradeoff)
    print('checkpoint path:', config.checkpoint)
    print('epochs:', config.epochs)
    print('learning rate:', config.lr)
    print()

    ###################################################################################################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('- Trainset parameters -')
    trainset = SimtbDataset(config.data, train=True, print_params=True, 
                            normalization='global' if config.norm == 0 else 'voxelwise')
    print()

    print('- Testset parameters -')
    testset = SimtbDataset(config.data, train=False, print_params=True, 
                           normalization='global' if config.norm == 0 else 'voxelwise')
    print()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    model = Model(k_maps=config.k)
    model = model.to(device)
    if config.checkpoint is not None:
        model.load_state_dict(torch.load(config.checkpoint))

    optimizer = torch.optim.Adam(model.parameters(), config.lr)

    timestr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    outdir = './out/{}'.format(timestr)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for epoch in range(config.epochs):
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(outdir, 'e{}.pt'.format(epoch)))

        model.train()
        train_loss = 0
        for i, data in enumerate(trainloader):
            mri_t, mri_v, mask = data
            mri_t = mri_t.float().to(device)
            mri_v = mri_v.float().to(device)
            mask = mask.bool().to(device)

            for n in range(mri_t.shape[0]):
                optimizer.zero_grad()

                model_out = model(mri_t[n], mask[n])

                mask_flat = torch.flatten(mask[n])
                in_flat = torch.reshape(mri_v[n], (mri_v[n].shape[0], -1))
                out_flat = torch.reshape(model_out, (model_out.shape[0], -1))

                in_masked = in_flat[:, mask_flat]
                out_masked = out_flat[:, mask_flat]

                if config.mode == 0:
                    loss = clustering_loss(out_masked, in_masked)
                else:
                    # loss = lstsq_loss(out_masked.t(), in_masked.t()) + config.tradeoff * hoyer_loss(out_masked)
                    loss = lstsq_loss(out_masked.t(), in_masked.t()) + 1.0 * entropy_loss(out_masked)
                    if i % 20 == 0:
                        # print(config.tradeoff * hoyer_loss(out_masked))
                        print(entropy_loss(out_masked))
                
                loss.backward()
                
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                train_loss += loss.item()

                optimizer.step()

        model.eval()
        eval_loss = 0
        for i, data in enumerate(testloader):
            mri_t, mri_v, mask = data
            mri_t = mri_t.float().to(device)
            mri_v = mri_v.float().to(device)
            mask = mask.bool().to(device)

            for n in range(mri_t.shape[0]):
                model_out = model(mri_t[n], mask[n])

                mask_flat = torch.flatten(mask[n])
                in_flat = torch.reshape(mri_v[n], (mri_v[n].shape[0], -1))
                out_flat = torch.reshape(model_out, (model_out.shape[0], -1))

                in_masked = in_flat[:, mask_flat]
                out_masked = out_flat[:, mask_flat]

                if config.mode == 0:
                    loss = clustering_loss(out_masked, in_masked)
                else:
                    # loss = lstsq_loss(out_masked.t(), in_masked.t()) + config.tradeoff * hoyer_loss(out_masked)
                    loss = lstsq_loss(out_masked.t(), in_masked.t()) + 1.0 * entropy_loss(out_masked)

                eval_loss += loss.item()

        print('[{}]\t\ttrain loss: {:.3f}\t\teval loss: {:.3f}'
              .format(epoch + 1, train_loss / len(trainloader.dataset), eval_loss / len(testloader.dataset)))

    torch.save(model.state_dict(), os.path.join(outdir, 'e{}.pt'.format(config.epochs)))
