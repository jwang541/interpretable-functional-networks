import torch



# Differentiable function to compute the projection matrix of the columns of Y
# onto the columnspace of X.
def lstsq_solution(x, y):
    return torch.mm(
        torch.pinverse(torch.mm(x.t(), x)),
        torch.mm(x.t(), y))


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


# Clustering loss (for pretraining)
def clustering_loss(x, y, eps=1e-8):
    spatial_mass = torch.sum(x, dim=1)
    spatial_density = torch.einsum('ks, k -> ks', x, 1.0 / (spatial_mass + eps))

    tcs = torch.einsum('ts, ks -> tk', y, spatial_density)
    y_recon = torch.einsum('tk, ks -> ts', tcs, x)

    return torch.sum(torch.square(y_recon - y))
