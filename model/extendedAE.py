"""
Defines 'Extended Auto-Encoders', which are basically spectrogram VAEs with an additional neural network
which infers synth parameters values from latent space values.
"""
import torch
import torch.nn as nn


class MLPExtendedAE(nn.Module):
    """ Model based on any compatible Auto-Encoder, with an additional non-invertible MLP regression model
    to infer synth parameters values. """

    def __init__(self, ae_model, architecture, dim_z, nb_params, dropout_p=0.0):
        super().__init__()
        self.ae_model = ae_model
        self.architecture = architecture
        # MLP automatically build from architecture string. E.g. '3l1024' means 3 hidden layers of 1024 neurons
        # Some options can be given after the underscore (e.g. '3l1024_nobn' adds the no batch norm argument)
        arch = self.architecture.split('_')  # This split might be useless
        if len(arch) == 1:
            num_hidden_layers, num_hidden_neurons = arch[0].split('l')
            num_hidden_layers, num_hidden_neurons = int(num_hidden_layers), int(num_hidden_neurons)
        else:
            raise NotImplementedError("Arch suffix arguments not implemented yet")
        self.mlp = nn.Sequential()
        for l in range(0, num_hidden_layers):
            if l == 0:
                self.mlp.add_module('fc{}'.format(l + 1), nn.Linear(dim_z, num_hidden_neurons))
            else:
                self.mlp.add_module('fc{}'.format(l+1), nn.Linear(num_hidden_neurons, num_hidden_neurons))
            # No dropout on the last layer before regression layer. TODO test remove dropout on 1st hidden layer
            if l < (num_hidden_layers - 1):
                self.mlp.add_module('drp{}'.format(l+1), nn.Dropout())
            self.mlp.add_module('act{}'.format(l+1), nn.ReLU())
        self.mlp.add_module('fc{}'.format(num_hidden_layers+1), nn.Linear(num_hidden_neurons, nb_params))
        self.mlp.add_module('reg_act', nn.Sigmoid())

    def forward(self, x):
        """
        Auto-encodes the input, and performs synth parameters regression.

        :returns: z_mu_logvar, z_sampled, x_out, u_out (synth params values)
        """
        z_mu_logvar, z_sampled, x_out = self.ae_model(x)
        u_out = self.mlp(z_sampled)  # z_sampled is z_mu during evaluation
        return z_mu_logvar, z_sampled, x_out, u_out

