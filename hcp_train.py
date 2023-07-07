import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn

from datasets import HcpDataset
from models import Model
from utils import *
from config import *

import resource




if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    train_mode_args = parser.add_mutually_exclusive_group(required=True)
    train_mode_args.add_argument('-p', '--pretrain', action='store_true', help='pretrain the model')
    train_mode_args.add_argument('-f', '--finetune', action='store_true', help='finetune the model')
    args = parser.parse_args()

    if args.pretrain:
        config = hcp_pretrain_config()
    else:
        config = hcp_finetune_config()

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

    ###################################################################################################################

    print('- Trainset parameters -')
    trainset = HcpDataset(data_index=config.data,
                          mask_file=config.mask,
                          print_params=True,
                          normalization='global' if config.norm == 0 else 'voxelwise',
                          crop_s=config.crop_s, crop_e=config.crop_e)
    print()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

    ###################################################################################################################

    model = Model(k_maps=config.k)
    model = model.to(device)
    if config.checkpoint is not None:
        model.load_state_dict(torch.load(config.checkpoint))

    ###################################################################################################################

    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    ###################################################################################################################

    timestr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    outdir = './out/{}'.format(timestr)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ###################################################################################################################

    for epoch in range(config.epochs):
        if epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(outdir, 'e{}.pt'.format(epoch)))

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

                if config.mode == 0:
                    loss = clustering_loss(out_masked, in_masked)
                else:
                    loss = lstsq_loss(out_masked.t(), in_masked.t()) + config.tradeoff * hoyer_loss(out_masked)
                
                loss.backward()
                # print('backward', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 'GB')

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                train_loss += loss.item()

                optimizer.step()

                # print('optimizer', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 'GB')
                # exit()

        print(epoch, train_loss)

    torch.save(model.state_dict(), os.path.join(outdir, 'e{}.pt'.format(config.epochs)))
