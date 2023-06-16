import os
import argparse

import torch
import torch.nn as nn

from datasets import HcpDataset
from models import Model
from utils import *





if __name__ == '__main__':

    # extract cmd args

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

    model = Model(k_maps=17)
    model = model.to(device)

    ###################################################################################################################

    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    ###################################################################################################################

    if not os.path.exists('./out'):
        os.makedirs('./out')

    ###################################################################################################################

    for i, data in enumerate(trainloader):
        mri, mask = data
        print(mri.shape, mask.shape)

    # p = 0.01

    # # for epoch in range(args.epochs):
    # for epoch in range(10):
    #     # if epoch % 10 == 0:
    #         # torch.save(model.state_dict(), './out/e{}.pt'.format(epoch))
    #     torch.save(model.state_dict(), './out/e{}.pt'.format(epoch))

    #     model.train()
    #     train_loss = 0
    #     for i, data in enumerate(trainloader):
    #         mri, mask = data
    #         mri = mri.float().to(device)
    #         mask = mask.bool().to(device)

    #         for n in range(mri.shape[0]):
    #             optimizer.zero_grad()

    #             sample = []
    #             for t in range(mri[n].shape[0]):
    #                 if random.random() < p:
    #                     sample.append(mri[n, t])

    #             # model_in = torch.stack(sample)
    #             model_in = mri[n, 0]
    #             model_in = model_in.unsqueeze(0)
    #             print('model in', model_in.shape)

    #             # model_in = mri[n] * mask[n]
    #             # model_in, rician_indices = add_rician_noise(model_in, mask[n], 0.25, std=0.25)
    #             # model_in, affine_indices = add_affine2d_noise(model_in, mask[n], 0.25, max_trans=0.05, max_angle=5.0)

    #             # Compute FNs with added noise
    #             model_out = model(model_in, mask[n])

    #             mask_flat = torch.flatten(mask[n])
    #             in_flat = torch.reshape(model_in, (model_in.shape[0], -1))
    #             out_flat = torch.reshape(model_out, (model_out.shape[0], -1))

    #             in_masked = in_flat[:, mask_flat]
    #             out_masked = out_flat[:, mask_flat]

    #             # Compute loss function without added noise; encourages model to be robust to noise
    #             print(lstsq_loss(out_masked.t(), in_masked.t()), hoyer_loss(out_masked))

    #             loss = lstsq_loss(out_masked.t(), in_masked.t()) + 10.0 * hoyer_loss(out_masked)

    #             # if args.finetune:
    #             #     loss = lstsq_loss(out_masked.t(), in_masked.t()) + args.trade_off * hoyer_loss(out_masked)
    #             # else:
    #             #     loss = clustering_loss(out_masked, in_masked)
                
    #             loss.backward()
                
    #             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #             train_loss += loss.item()

    #             optimizer.step()

    #     print(epoch, train_loss)
