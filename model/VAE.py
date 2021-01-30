
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import contextlib
from torch.autograd import profiler


class BasicVAE(nn.Module):
    """ A standard VAE that uses some given encoder and decoder networks.
     The latent probability distribution is modeled as dim_z independent Gaussian distributions. """

    def __init__(self, encoder, dim_z, decoder):
        super().__init__()
        # No size checks performed. Encoder and decoder must have been properly designed
        self.encoder = encoder
        self.dim_z = dim_z
        self.decoder = decoder
        # All VAEs must always store encoded values and the sampled latent vector
        self.z_mu_logsigma2 = None
        self.z_sampled = None
        self.is_profiled = False

    def forward(self, x):
        """ Encodes the given input into a Qphi(z|x) probability distribution, stores encoded distribution
        parameters, samples a latent vector from that distribution, and finally calls the decoder network.

        Encoded distributions and sampled latent vector are stored in class attributes.
        """
        with profiler.record_function("ENCODING") if self.is_profiled else contextlib.nullcontext():
            # TODO changer ça - ça doit devenir une sortie pour être OK avec DataParallel
            self.z_mu_logsigma2 = self.encoder(x)
            n_minibatch = self.z_mu_logsigma2.size()[0]
            mu = self.z_mu_logsigma2[:, 0, :]
            sigma = torch.exp(self.z_mu_logsigma2[:, 1, :] / 2.0)
        with profiler.record_function("LATENT_SAMPLING") if self.is_profiled else contextlib.nullcontext():
            if self.training:
                # Sampling from the Qphi(z|x) probability distribution - with re-parametrization trick
                eps = Normal(torch.zeros(n_minibatch, self.dim_z, device=mu.device),
                             torch.ones(n_minibatch, self.dim_z, device=mu.device)).sample()
                self.z_sampled = mu + sigma * eps
            else:  # eval mode: no random sampling
                self.z_sampled = mu
        with profiler.record_function("DECODING") if self.is_profiled else contextlib.nullcontext():
            x_out = self.decoder(self.z_sampled)
        return x_out

