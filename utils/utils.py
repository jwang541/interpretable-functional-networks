import torch
import torch.linalg as linalg

import random


# Differentiable function to compute the projection matrix of the columns of Y
# onto the columnspace of X.
def lstsq_solution(x, y):
    return torch.mm(torch.pinverse(x), y)


# Differentiable function to compute the lstsq residuals of the projection of 
# the columns of Y onto the columnspace of X.
def lstsq_loss(x, y):
    soln = lstsq_solution(x, y)
    y_recon = torch.mm(x, soln)
    return torch.sum(torch.square(y_recon - y))


# Hoyer sparsity term
def hoyer_loss(x, eps=1e-8):
    return torch.sum(
        torch.sum(
            torch.divide(
                torch.sum(torch.abs(x), dim=1) + 1,
                torch.sqrt(torch.sum(torch.square(x), dim=1) + eps))))


def entropy_loss(x):
    x = x + 1e-8
    x = x / torch.sum(x, dim=0)
    return -1 * torch.sum(x * torch.log(x))


# Clustering loss (for pretraining)
def clustering_loss(x, y, eps=1e-8):
    spatial_mass = torch.sum(x, dim=1)
    spatial_density = torch.einsum('ks, k -> ks', x, 1.0 / (spatial_mass + eps))
    tcs = torch.einsum('ts, ks -> tk', y, spatial_density)
    y_recon = torch.einsum('tk, ks -> ts', tcs, x)
    return torch.sum(torch.square(y_recon - y))


# Add rician noise with standard deviation std to each time point with probability p
# and return indices which had noise added
def add_rician_noise(x, mask, p, std):
    with torch.no_grad():
        noisy_x = x.clone()
        noisy_indices = []

        for t in range(x.shape[0]):
            if random.random() < p:
                noise1 = std * torch.randn_like(noisy_x[t])
                noise2 = std * torch.randn_like(noisy_x[t])
                noisy_x[t] = torch.sqrt(torch.square(noisy_x[t] + noise1) + torch.square(noise2))
                noisy_indices.append(t)

        return noisy_x * mask, noisy_indices


# Apply a random affine transformation to each time point with probability p and
# return indices which had noise added
def add_affine2d_noise(x, mask, p, max_trans, max_angle):
    with torch.no_grad():
        noisy_x = x.clone()
        noisy_indices = []

        for t in range(x.shape[0]):
            if random.random() < p:
                affine = transforms.RandomAffine(degrees=max_angle, translate=(max_trans, max_trans))
                noisy_x[t] = affine(noisy_x[t])
                noisy_indices.append(t)

        return noisy_x * mask, noisy_indices
