"""
Defines 'Extended Auto-Encoders', which are basically spectrogram VAEs with an additional neural network
which infers synth parameters values from latent space values.
"""

import torch.nn as nn

from model import VAE
import model.regression
from data.preset import PresetIndexesHelper


class ExtendedAE(nn.Module):
    """ Model based on any compatible Auto-Encoder and Regression models. """

    def __init__(self, ae_model, reg_model, idx_helper: PresetIndexesHelper, dropout_p=0.0):
        super().__init__()
        self.idx_helper = idx_helper  # unused at the moment
        self.ae_model = ae_model
        if isinstance(self.ae_model, VAE.BasicVAE):
            self._is_flow_based_latent_space = False
        elif isinstance(self.ae_model, VAE.FlowVAE):
            self._is_flow_based_latent_space = True
        else:
            raise TypeError("Unrecognized auto-encoder model")
        self.reg_model = reg_model
        if isinstance(self.reg_model, model.regression.FlowRegression):
            self._is_flow_based_regression = True
        elif isinstance(self.reg_model, model.regression.MLPRegression):
            self._is_flow_based_regression = False
        else:
            raise TypeError("Unrecognized synth params regression model")

    @property
    def is_flow_based_latent_space(self):
        return self._is_flow_based_latent_space

    @property
    def is_flow_based_regression(self):
        return self._is_flow_based_regression

    def forward(self, x):
        """
        Auto-encodes the input (does NOT perform synth parameters regression).
        This class must not store temporary self.* tensors for e.g. loss computation, because it will
        be parallelized on multiple GPUs, and output tensors will be concatenated dy DataParallel.
        """
        return self.ae_model(x)

    def latent_loss(self, z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac):
        return self.ae_model.latent_loss(z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac)

