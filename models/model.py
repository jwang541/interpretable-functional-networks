import torch
import torch.nn as nn
import torch.nn.functional as F

from .parts import UNet, SqueezeExcitation3D, SCSqueezeExcitation3D


    
# Takes fMRI data of dimension (T, D, H, W) and an optional mask of broadcastable dimension,
# and produces a feature map of dimension (K, D, H, W).
class Model(nn.Module):
    def __init__(self, k_maps, eps=1e-8, debug=False):
        super().__init__()
        self.k_maps = k_maps
        self.eps = eps

        self.unet = UNet(k_maps=16, eps=eps)
        self.conv_out = nn.Conv3d(16, k_maps, kernel_size=3, stride=1, padding=1, bias=False)
        self.attention = SqueezeExcitation3D(in_channels=k_maps, channels=16, out_activation='sigmoid', debug=debug)

        nn.init.trunc_normal_(self.conv_out.weight, std=0.01, a=-0.02, b=0.02)

        self.debug = debug

    def forward(self, x, mask=None):
        x = x * mask
        x = x.unsqueeze(1)

        x = self.unet(x)
        x = F.relu(self.conv_out(x))
        x = self.attention(x)
        x = torch.mean(x, dim=0)
        x = x * mask

        component_max = torch.amax(x, dim=(1, 2, 3))
        x = torch.einsum('kxyz, k -> kxyz', x, 1.0 / (component_max + self.eps))

        return x
