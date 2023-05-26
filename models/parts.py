import torch
import torch.nn as nn
import torch.nn.functional as F



# Implements the scSE (spatial and channel Squeeze-and-Excitation) block proposed in "Recalibrating Fully
# Convolutional Networks with Spatial and Channel ‘Squeeze & Excitation’ Blocks" (Roy et al.) for an input
# with 3 spatial dimensions.
class SCSqueezeExcitation3D(nn.Module):
    def __init__(self, in_channels, channel_neurons):
        super().__init__()
        self.in_channels = in_channels

        self.channel_se = nn.Sequential(
            nn.Linear(in_channels, channel_neurons, bias=True),
            nn.ReLU(),
            nn.Linear(channel_neurons, in_channels, bias=True),
            nn.Sigmoid()
        )
        self.spatial_se = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, X):
        Z_channel = torch.mean(X, dim=(2, 3, 4))
        S_channel = self.channel_se(Z_channel)
        S_spatial = self.spatial_se(X)

        X_channel = torch.einsum('ncxyz, nc -> ncxyz', X, S_channel)
        X_spatial = torch.mul(X, S_spatial)

        return torch.max(X_channel, X_spatial)
    


# Implements a UNet module (Ronneberger et al.).
class UNet(nn.Module):
    def __init__(self, k_maps, eps=1e-9):
        super().__init__()
        self.k_maps = k_maps
        self.eps = eps

        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.deconv5 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv6 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv7 = nn.Conv3d(64, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv9 = nn.Conv3d(16, k_maps, kernel_size=3, stride=1, padding=1, bias=False)

        nn.init.trunc_normal_(self.conv1.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.conv2.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.conv3.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.conv4.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.deconv5.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.deconv6.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.deconv7.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.conv8.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.conv9.weight, std=0.01, a=-0.02, b=0.02)

        self.norm1 = nn.InstanceNorm3d(16, affine=True)
        self.norm2 = nn.InstanceNorm3d(32, affine=True)
        self.norm3 = nn.InstanceNorm3d(32, affine=True)
        self.norm4 = nn.InstanceNorm3d(32, affine=True)
        self.norm5 = nn.InstanceNorm3d(32, affine=True)
        self.norm6 = nn.InstanceNorm3d(32, affine=True)
        self.norm7 = nn.InstanceNorm3d(16, affine=True)
        self.norm8 = nn.InstanceNorm3d(16, affine=True)
        self.norm9 = nn.InstanceNorm3d(k_maps, affine=True)

    def forward(self, x):
        x1 = self.norm1(F.leaky_relu(self.conv1(x), negative_slope=0.01))
        x2 = self.norm2(F.leaky_relu(self.conv2(x1), negative_slope=0.01))
        x3 = self.norm3(F.leaky_relu(self.conv3(x2), negative_slope=0.01))
        x4 = self.norm4(F.leaky_relu(self.conv4(x3), negative_slope=0.01))
        x5 = self.norm5(F.leaky_relu(self.deconv5(x4), negative_slope=0.01))
        x5 = torch.cat([F.interpolate(x5, x3.shape[2:5]), x3], 1)
        x6 = self.norm6(F.leaky_relu(self.deconv6(x5), negative_slope=0.01))
        x6 = torch.cat([F.interpolate(x6, x2.shape[2:5]), x2], 1)
        x7 = self.norm7(F.leaky_relu(self.deconv7(x6), negative_slope=0.01))
        x7 = torch.cat([F.interpolate(x7, x1.shape[2:5]), x1], 1)
        x8 = self.norm8(F.leaky_relu(self.conv8(x7), negative_slope=0.01))
        x9 = self.norm9(F.leaky_relu(self.conv9(x8), negative_slope=0.01))
        return x9