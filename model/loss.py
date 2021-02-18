
import torch
import torch.nn as nn


# TODO Spectral Convergence


class GaussianDkl:
    """ Kullback-Leibler Divergence between independant Gaussian distributions (diagonal
    covariance matrices). mu 2 and logs(var) 2 are optional and will be resp. zeros and ones if not given.

    A normalization over the batch dimension will automatically be performed.
    An optional normalization over the channels dimension can also be performed.

    All tensor sizes should be (N_minibatch, N_channels) """
    def __init__(self, normalize=True):
        self.normalize = normalize  # Normalization over channels

    def __call__(self, mu1, logvar1, mu2=None, logvar2=None):
        if mu2 is None and logvar2 is None:
            Dkl = 0.5 * torch.sum(torch.exp(logvar1) + torch.square(mu1) - logvar1 - 1.0)
        else:
            raise NotImplementedError("General Dkl not implemented yet...")
        Dkl = Dkl / mu1.size(0)
        if self.normalize:
            return Dkl / mu1.size(1)
        else:
            return Dkl


# TODO MMD


# TODO DX7 parameters Loss: to get a meaningful (but non-differentiable) loss, inferred parameter values must be
#    quantized as they would be in Dexed

