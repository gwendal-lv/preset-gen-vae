"""
Neural networks classes for synth parameters regression, and related utility functions.

These regression models can be used on their own, or passed as constructor arguments to
extended AE models.
"""

from collections.abc import Iterable

import torch.nn as nn

from data.preset import PresetIndexesHelper


class PresetActivation(nn.Module):
    """ Applies the appropriate activations (e.g. sigmoid, hardtanh, softmax, ...) to different neurons
    or groups of neurons of a given input layer. """
    def __init__(self, idx_helper: PresetIndexesHelper,
                 numerical_activation=nn.Hardtanh(min_val=0.0, max_val=1.0)):
        """

        :param idx_helper:
        :param numerical_activation: Should be nn.Hardtanh if numerical params often reach 0.0 and 1.0 GT values,
            or nn.Sigmoid to perform a smooth regression without extreme 0.0 and 1.0 values.
        """
        super().__init__()
        self.idx_helper = idx_helper
        self.numerical_act = numerical_activation
        self.categorical_act = nn.Softmax(dim=-1)  # Required for categorical cross-entropy loss
        # Pre-compute indexes lists (to use less CPU)
        self.num_indexes = self.idx_helper.get_numerical_learnable_indexes()
        self.cat_indexes = self.idx_helper.get_categorical_learnable_indexes()  # type: Iterable[Iterable]

    def forward(self, x):
        """ Applies per-parameter output activations using the PresetIndexesHelper attribute of this instance. """
        x[:, self.num_indexes] = self.numerical_act(x[:, self.num_indexes])
        for cat_learnable_indexes in self.cat_indexes:  # type: Iterable
            x[:, cat_learnable_indexes] = self.categorical_act(x[:, cat_learnable_indexes])
        return x


class MLPRegression(nn.Module):

    def __init__(self, architecture, dim_z, idx_helper: PresetIndexesHelper, dropout_p = 0.0):
        """

        :param architecture: MLP automatically built from architecture string. E.g. '3l1024' means
            3 hidden layers of 1024 neurons. Some options can be given after the underscore
            (e.g. '3l1024_nobn' adds the no batch norm argument).
        :param dim_z: Size of a z_K latent vector
        :param idx_helper:
        :param dropout_p:
        """
        super().__init__()
        self.architecture = architecture.split('_')  # Split between base args and opt args (e.g. _nobn)
        self.dim_z = dim_z
        self.idx_helper = idx_helper
        if len(self.architecture) == 1:
            num_hidden_layers, num_hidden_neurons = self.architecture[0].split('l')
            num_hidden_layers, num_hidden_neurons = int(num_hidden_layers), int(num_hidden_neurons)
        else:
            raise NotImplementedError("Arch suffix arguments not implemented yet")
        # Layers definition
        self.reg_model = nn.Sequential()
        for l in range(0, num_hidden_layers):
            if l == 0:
                self.reg_model.add_module('fc{}'.format(l + 1), nn.Linear(dim_z, num_hidden_neurons))
            else:
                self.reg_model.add_module('fc{}'.format(l + 1), nn.Linear(num_hidden_neurons, num_hidden_neurons))
            # No dropout on the last layer before regression layer. TODO test remove dropout on 1st hidden layer
            if l < (num_hidden_layers - 1):
                self.reg_model.add_module('drp{}'.format(l + 1), nn.Dropout(dropout_p))
            self.reg_model.add_module('act{}'.format(l + 1), nn.ReLU())
        self.reg_model.add_module('fc{}'.format(num_hidden_layers + 1), nn.Linear(num_hidden_neurons,
                                                                                  self.idx_helper.learnable_preset_size))
        # dedicated activation module - because we need a per-parameter activation (e.g. sigmoid or softmax)
        self.reg_model.add_module('act', PresetActivation(self.idx_helper))

    def forward(self, z_K):
        """ Applies the regression model to a z_K latent vector (VAE latent flow output samples). """
        return self.reg_model(z_K)




