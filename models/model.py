import torch
import torch.nn as nn
import torch.nn.functional as F

from .parts import UNet, SqueezeExcitation3D, SCSqueezeExcitation3D



# Takes fMRI data of dimension (T, D, H, W) and an optional mask of dimension (D, H, W),
# and produces a feature map of dimension (K, D, H, W).
class Model(nn.Module):
    def __init__(self, k_maps, eps=1e-8, debug=False):
        super().__init__()
        self.k = k_maps
        self.eps = eps
        self.debug = debug

        # UNet weights
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
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
        self.proj = nn.Conv3d(32, 8, kernel_size=1, stride=1, padding=0, bias=True)
        self.attention = nn.Sequential(
            nn.Linear(8, 8, bias=True),
            nn.ReLU(),
            nn.Linear(8, self.k, bias=True),
            nn.Sigmoid()
        )
        self.last_attention = None

    def forward(self, x, mask):
        x = x * mask
        x = x.unsqueeze(1)

        # Apply UNet to estimate functional networks
        x1 = self.norm1(F.leaky_relu(self.conv1(x), negative_slope=0.01))
        x2 = self.norm2(F.leaky_relu(F.max_pool3d(self.conv2(x1), 2, ceil_mode=True), negative_slope=0.01))
        x3 = self.norm3(F.leaky_relu(F.max_pool3d(self.conv3(x2), 2, ceil_mode=True), negative_slope=0.01))
        x4 = self.norm4(F.leaky_relu(F.max_pool3d(self.conv4(x3), 2, ceil_mode=True), negative_slope=0.01)) # x4 is bottleneck layer
        x5 = self.norm5(F.leaky_relu(self.deconv5(x4), negative_slope=0.01))
        x5 = torch.cat([F.interpolate(x5, x3.shape[2:5]), x3], 1)
        x6 = self.norm6(F.leaky_relu(self.deconv6(x5), negative_slope=0.01))
        x6 = torch.cat([F.interpolate(x6, x2.shape[2:5]), x2], 1)
        x7 = self.norm7(F.leaky_relu(self.deconv7(x6), negative_slope=0.01))
        x7 = torch.cat([F.interpolate(x7, x1.shape[2:5]), x1], 1)
        x8 = self.norm8(F.leaky_relu(self.conv8(x7), negative_slope=0.01))
        maps = F.relu(self.conv9(x8))

        # Mask and normalize UNet maps so they have a mean of 1.0
        maps = maps * mask
        maps = maps / (torch.mean(maps, dim=(2, 3, 4), keepdim=True) + self.eps)

        # Extract attention weights from bottleneck layer
        proj = F.relu(self.proj(x4))
        scores = torch.mean(proj, dim=(2, 3, 4))
        weights = self.attention(scores)
        if self.debug:
            self.last_attention = weights

        # Weight normalized maps with attention values, then average to get combined maps
        y = torch.einsum('tk, tkxyz -> tkxyz', weights, maps)
        y = torch.mean(y, dim=0)

        # Divide by means so each map has mean intensity 1
        y = y / (torch.mean(y, dim=(1, 2, 3), keepdim=True) + self.eps)
        return y

