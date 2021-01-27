
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


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

    def forward(self, x):
        """ Encodes the given input into a Qphi(z|x) probability distribution, stores encoded distribution
        parameters, samples a latent vector from that distribution, and finally calls the decoder network.

        Encoded distributions and sampled latent vector are stored in class attributes.
        """
        self.z_mu_logsigma2 = self.encoder(x)
        n_minibatch = self.z_mu_logsigma2.size()[0]
        mu = self.z_mu_logsigma2[:, 0, :]
        sigma = torch.exp(self.z_mu_logsigma2[:, 1, :] / 2.0)
        if self.training:
            # Sampling from the Qphi(z|x) probability distribution - with re-parametrization trick
            eps = Normal(torch.zeros(n_minibatch, self.dim_z), torch.ones(n_minibatch, self.dim_z)).sample()
            self.z_sampled = mu + eps * sigma
        else:  # eval mode: no random sampling
            self.z_sampled = mu
        return self.decoder(self.z_sampled)
