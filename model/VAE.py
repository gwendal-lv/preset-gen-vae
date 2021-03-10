
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import contextlib
from torch.autograd import profiler

from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

import model.loss
from utils.probability import gaussian_log_probability, standard_gaussian_log_probability


class BasicVAE(nn.Module):
    """ A standard VAE that uses some given encoder and decoder networks.
     The latent probability distribution is modeled as dim_z independent Gaussian distributions. """

    def __init__(self, encoder, dim_z, decoder, normalize_latent_loss, latent_loss_type):
        super().__init__()
        # No size checks performed. Encoder and decoder must have been properly designed
        self.encoder = encoder
        self.dim_z = dim_z
        self.decoder = decoder
        self.is_profiled = False
        if latent_loss_type.lower() == 'dkl':
            # TODO try don't normalize (if reconstruction loss is not normalized either)
            self.latent_criterion = model.loss.GaussianDkl(normalize=normalize_latent_loss)
        else:
            raise NotImplementedError("Latent loss '{}' unavailable".format(latent_loss_type))

    def forward(self, x):
        """ Encodes the given input into a q_phi(z|x) probability distribution,
        samples a latent vector from that distribution, and finally calls the decoder network.

        For compatibility, it returns zK_sampled = z_sampled and the log abs det jacobian(T) = 0.0
        (T = identity)

        :returns: z_mu_logvar, z_sampled, zK_sampled=z_sampled, logabsdetjacT=0.0, x_out (reconstructed spectrogram)
        """
        with profiler.record_function("ENCODING") if self.is_profiled else contextlib.nullcontext():
            z_mu_logvar = self.encoder(x)
            n_minibatch = z_mu_logvar.size()[0]
            mu = z_mu_logvar[:, 0, :]
            sigma = torch.exp(z_mu_logvar[:, 1, :] / 2.0)
        with profiler.record_function("LATENT_SAMPLING") if self.is_profiled else contextlib.nullcontext():
            if self.training:
                # Sampling from the q_phi(z|x) probability distribution - with re-parametrization trick
                eps = Normal(torch.zeros(n_minibatch, self.dim_z, device=mu.device),
                             torch.ones(n_minibatch, self.dim_z, device=mu.device)).sample()
                z_sampled = mu + sigma * eps
            else:  # eval mode: no random sampling
                z_sampled = mu
        with profiler.record_function("DECODING") if self.is_profiled else contextlib.nullcontext():
            x_out = self.decoder(z_sampled)
        return z_mu_logvar, z_sampled, z_sampled, torch.zeros((z_sampled.shape[0], 1), device=x.device), x_out

    def latent_loss(self, z_0_mu_logvar, **kwargs):
        """ **kwargs are not used (they exist for compatibility with flow-based latent spaces). """
        # Default: divergence or discrepancy vs. zero-mean unit-variance multivariate gaussian
        return self.latent_criterion(z_0_mu_logvar[:, 0, :], z_0_mu_logvar[:, 1, :])


class FlowVAE(nn.Module):
    """
    A VAE with flow transforms in the latent space.
    q_ZK(z_k) is a complex distribution and does not have a closed-form expression.

    The loss does not rely on a Kullback-Leibler divergence but on a direct log-likelihood computation.
    """

    def __init__(self, encoder, dim_z, decoder, normalize_latent_loss,
                 flow_layers, flow_hidden_features,
                 # TODO add more flow params (hidden neural networks config: BN, layers, ...)
                 ):
        # TODO reserve midi pitch/vel latent variables? add ctor argument
        super().__init__()
        # No size checks performed. Encoder and decoder must have been properly designed
        self.encoder = encoder
        self.dim_z = dim_z
        self.decoder = decoder
        self.is_profiled = False
        # Latent flow setup
        self.normalize_latent_loss = normalize_latent_loss
        self.flow_layers_count = flow_layers
        self.flow_hidden_features = flow_hidden_features
        transforms = []
        for _ in range(self.flow_layers_count):
            transforms.append(ReversePermutation(features=self.dim_z))
            transforms.append(MaskedAffineAutoregressiveTransform(features=self.dim_z,
                                                                  hidden_features=self.flow_hidden_features))
        self.flow_transform = CompositeTransform(transforms)

    def forward(self, x):
        """ Encodes the given input into a q_Z0(z_0|x) probability distribution,
        samples a latent vector from that distribution,
        transforms it into q_ZK(z_K|x) using a invertible normalizing flow,
        and finally calls the decoder network using the z_K samples.

        :returns: z0_mu_logvar, z0_sampled, zK_sampled, logabsdetjacT, x_out (reconstructed spectrogram)
        """
        with profiler.record_function("ENCODING") if self.is_profiled else contextlib.nullcontext():
            z_0_mu_logvar = self.encoder(x)
            n_minibatch = z_0_mu_logvar.size()[0]
            mu0 = z_0_mu_logvar[:, 0, :]
            sigma0 = torch.exp(z_0_mu_logvar[:, 1, :] / 2.0)
        with profiler.record_function("LATENT_FLOW") if self.is_profiled else contextlib.nullcontext():
            if self.training:
                # Sampling from the q_phi(z|x) probability distribution - with re-parametrization trick
                eps = Normal(torch.zeros(n_minibatch, self.dim_z, device=mu0.device),
                             torch.ones(n_minibatch, self.dim_z, device=mu0.device)).sample()
                z_0_sampled = mu0 + sigma0 * eps
            else:  # eval mode: no random sampling
                z_0_sampled = mu0
            # Forward flow (fast with nflows MAF implementation)
            z_K_sampled, log_abs_det_jac = self.flow_transform(z_0_sampled)
        with profiler.record_function("DECODING") if self.is_profiled else contextlib.nullcontext():
            x_out = self.decoder(z_K_sampled)
        return z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac , x_out

    def latent_loss(self, z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac):
        # log-probability of z_0 is evaluated knowing the gaussian distribution it was sampled from
        log_q_Z0_z0 = gaussian_log_probability(z_0_sampled, z_0_mu_logvar[:, 0, :], z_0_mu_logvar[:, 1, :])
        # log-probability of z_K in the prior p_theta distribution
        # We model this prior as a zero-mean unit-variance multivariate gaussian
        log_p_theta_zK = standard_gaussian_log_probability(z_K_sampled)
        # Returned is the opposite of the ELBO terms
        if not self.normalize_latent_loss:  # Default, which returns actual ELBO terms
            return -(log_p_theta_zK - log_q_Z0_z0 + log_abs_det_jac).mean()  # Mean over batch dimension
        else:  # Mean over batch dimension and latent vector dimension (D)
            return -(log_p_theta_zK - log_q_Z0_z0 + log_abs_det_jac).mean() / z_0_sampled.shape[1]

