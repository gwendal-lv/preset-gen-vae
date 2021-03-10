"""
Utility functions related to probabilities and statistics, e.g. log likelihoods, ...
"""

import numpy as np

import torch


__log_2_pi = np.log(2*np.pi)


def standard_gaussian_log_probability(samples):
    """
    Computes the log-probabilities of given batch of samples using a multivariate gaussian distribution
    of independent components (zero-mean, identity covariance matrix).
    """
    return -0.5 * (samples.shape[1] * __log_2_pi + torch.sum(samples**2, dim=1))


def gaussian_log_probability(samples, mu, log_var):
    """
    Computes the log-probabilities of given batch of samples using a multivariate gaussian distribution
    of independent components (diagonal covariance matrix).
    """
    # if samples and mu do not have the same size,
    # torch automatically properly performs the subtract if mu is 1 dim smaller than samples
    return -0.5 * (samples.shape[1] * __log_2_pi +
                   torch.sum( log_var + ((samples - mu)**2 / torch.exp(log_var)), dim=1))



