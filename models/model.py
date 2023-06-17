import torch
import torch.nn as nn
import torch.nn.functional as F

from .parts import UNet, SqueezeExcitation3D, SCSqueezeExcitation3D



class Model(nn.Module):
    def __init__(self, k_maps, eps=1e-8, debug=False):
        super().__init__()
        self.k = k_maps
        self.eps = eps

        # UNet weights
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.deconv5 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv6 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv7 = nn.Conv3d(64, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv9 = nn.Conv3d(16, self.k, kernel_size=3, stride=1, padding=1, bias=False)

        self.norm1 = nn.InstanceNorm3d(16, affine=True)
        self.norm2 = nn.InstanceNorm3d(32, affine=True)
        self.norm3 = nn.InstanceNorm3d(32, affine=True)
        self.norm4 = nn.InstanceNorm3d(32, affine=True)
        self.norm5 = nn.InstanceNorm3d(32, affine=True)
        self.norm6 = nn.InstanceNorm3d(32, affine=True)
        self.norm7 = nn.InstanceNorm3d(16, affine=True)
        self.norm8 = nn.InstanceNorm3d(16, affine=True)
        self.norm9 = nn.InstanceNorm3d(self.k, affine=True)

        # Spatial attention weights
        self.attention = nn.Sequential(
            nn.Linear(32, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, self.k, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        x = x * mask
        x = x.unsqueeze(1)

        # print('x', torch.isnan(x).any().item(), x.shape)

        # Apply UNet 
        x1 = self.norm1(F.leaky_relu(self.conv1(x), negative_slope=0.01))
        # print('x1', torch.isnan(x1).any().item())

        x2 = self.norm2(F.leaky_relu(self.conv2(x1), negative_slope=0.01))
        # print('x2', torch.isnan(x2).any().item())

        x3 = self.norm3(F.leaky_relu(self.conv3(x2), negative_slope=0.01))
        # print('x3', torch.isnan(x3).any().item())

        x4 = self.norm4(F.leaky_relu(self.conv4(x3), negative_slope=0.01)) # x4 is bottleneck layer
        # print('x4', torch.isnan(x4).any().item())

        x5 = self.norm5(F.leaky_relu(self.deconv5(x4), negative_slope=0.01))
        x5 = torch.cat([F.interpolate(x5, x3.shape[2:5]), x3], 1)
        x6 = self.norm6(F.leaky_relu(self.deconv6(x5), negative_slope=0.01))
        x6 = torch.cat([F.interpolate(x6, x2.shape[2:5]), x2], 1)
        x7 = self.norm7(F.leaky_relu(self.deconv7(x6), negative_slope=0.01))
        x7 = torch.cat([F.interpolate(x7, x1.shape[2:5]), x1], 1)
        x8 = self.norm8(F.leaky_relu(self.conv8(x7), negative_slope=0.01))
        maps = F.relu(self.conv9(x8))

        # print('maps', torch.isnan(maps).any().item())

        # Extract attention weights from bottleneck layer
        scores = torch.mean(x4, dim=(2, 3, 4))
        weights = self.attention(scores)

        # print('scores', torch.isnan(scores).any().item(), scores.shape)
        # print('weights', torch.isnan(weights).any().item())

        # Normalize UNet maps so they have a mean of 1.0
        maps_means = torch.mean(maps, dim=(2, 3, 4), keepdims=True)
        maps = maps / (maps_means + self.eps)

        # print('normal maps', torch.isnan(maps).any().item())

        # Weight normalized maps with attention values, then average to get combined maps
        y = torch.einsum('nk, nkxyz -> nkxyz', weights, maps)
        y = torch.mean(y, dim=0)
        # print('y', torch.isnan(y).any().item())
        # print()

        return y







# # Takes fMRI data of dimension (T, D, H, W) and an optional mask of dimension (D, H, W),
# # and produces a feature map of dimension (K, D, H, W).
# class Model(nn.Module):
#     def __init__(self, k_maps, eps=1e-8, debug=False):
#         super().__init__()
#         self.k_maps = k_maps
#         self.eps = eps

#         self.unet = UNet(k_maps=16, eps=eps)
#         self.conv_out = nn.Conv3d(16, k_maps, kernel_size=3, stride=1, padding=1, bias=False)
#         self.attention = SqueezeExcitation3D(in_channels=k_maps, channels=16, out_activation='sigmoid', debug=debug)

#         nn.init.trunc_normal_(self.conv_out.weight, std=0.01, a=-0.02, b=0.02)

#         self.debug = debug

#     def forward(self, x, mask=None):
#         x = x * mask
#         x = x.unsqueeze(1)

#         x = self.unet(x)
#         x = F.relu(self.conv_out(x))
#         x = self.attention(x)
#         x = torch.mean(x, dim=0)
#         x = x * mask

#         component_max = torch.amax(x, dim=(1, 2, 3))
#         x = torch.einsum('kxyz, k -> kxyz', x, 1.0 / (component_max + self.eps))

#         return x
