"""
Neural networks classes for synth parameters regression, and related utility functions.

These regression models can be used on their own, or passed as constructor arguments to
extended AE models.
"""

from collections.abc import Iterable

import torch.nn as nn

from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from data.preset import PresetIndexesHelper
from model.flows import CustomRealNVP, InverseFlow


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


# TODO class to "reverse" preset softmax activations.
#    could be done by using the properly one-hot encoded sub-vectors, by applying a simple affine functions
#    (whose coeffs will depend on the size of the one-hot sub-vector, to always get the same softmax activation)


class MLPRegression(nn.Module):
    def __init__(self, architecture, dim_z, idx_helper: PresetIndexesHelper, dropout_p=0.0):
        """
        :param architecture: MLP automatically built from architecture string. E.g. '3l1024' means
            3 hidden layers of 1024 neurons. Some options can be given after an underscore
            (e.g. '3l1024_nobn' adds the no batch norm argument). See implementation for more details.  TODO implement
        :param dim_z: Size of a z_K latent vector
        :param idx_helper:
        :param dropout_p:
        """
        super().__init__()
        self.architecture = architecture.split('_')  # Split between base args and opt args (e.g. _nobn)
        self.dim_z = dim_z
        self.idx_helper = idx_helper  # Useless here?
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


class FlowRegression(nn.Module):
    def __init__(self, architecture, dim_z, idx_helper: PresetIndexesHelper, dropout_p=0.0,
                 fast_forward_flow=True):
        """
        :param architecture: Flow automatically built from architecture string. E.g. 'realnvp_16l200' means
            16 RealNVP flow layers with 200 hidden features each. Some options can be given after an underscore
            (e.g. '16l200_bn' adds batch norm). See implementation for more details.  TODO implement suffix options
        :param dim_z: Size of a z_K latent vector, which is also the output size for this invertible normalizing flow.
        :param idx_helper:
        :param dropout_p:  TODO implement dropout (in all but the last flow layers)
        :param fast_forward_flow: If True, the flow transform will be built such that it is fast (and memory-efficient)
            in the forward direction (else, it will be fast in the inverse direction). Moreover, if batch-norm is used
            between layers, the flow can be trained only its 'fast' direction (which can be forward or inverse
            depending on this argument).
        """
        super().__init__()
        self.dim_z = dim_z
        self.idx_helper = idx_helper
        self._fast_forward_flow = fast_forward_flow
        arch_args = architecture.split('_')
        if len(arch_args) < 2:
            raise AssertionError("Unvalid architecture string argument '{}' does not contain enough information"
                                 .format(architecture))
        elif len(arch_args) == 2:  # No opt args (default)
            self.flow_type = arch_args[0]
            self.num_flow_layers, self.num_flow_hidden_features = arch_args[1].split('l')
            self.num_flow_layers = int(self.num_flow_layers)
            self.num_flow_hidden_features = int(self.num_flow_hidden_features)
            # Default: full BN usage
            self.bn_between_flows = True
            self.bn_within_flows = True
        else:
            raise NotImplementedError("Arch suffix arguments not implemented yet (too many arch args given in '{}')"
                                      .format(architecture))
        # Multi-layer flow definition
        if self.flow_type.lower() == 'realnvp' or self.flow_type.lower() == 'rnvp':
            # RealNVP - custom (without useless gaussian base distribution) and no BN on last layers
            self._forward_flow_transform = CustomRealNVP(self.dim_z, self.num_flow_hidden_features,
                                                         self.num_flow_layers,
                                                         num_blocks_per_layer=2,  # MAF default
                                                         batch_norm_between_layers=self.bn_between_flows,
                                                         batch_norm_within_layers=self.bn_within_flows,
                                                         dropout_probability=dropout_p
                                                         )
        elif self.flow_type.lower() == 'maf':
            transforms = []
            for l in range(self.num_flow_layers):
                transforms.append(ReversePermutation(features=self.dim_z))
                # TODO Batch norm added on all flow MLPs but the 2 last
                #     and dropout p
                transforms.append(MaskedAffineAutoregressiveTransform(features=self.dim_z,
                                                                      hidden_features=self.num_flow_hidden_features,
                                                                      use_batch_norm=False,  # TODO (l < num_layers-2),
                                                                      dropout_probability=0.5  # TODO as param
                                                                      ))
            # The inversed maf flow should never (cannot...) be used during training:
            #   - much slower than forward (in nflows implementation)
            #   - very unstable
            #   - needs ** huge ** amounts of GPU RAM
            self._forward_flow_transform = CompositeTransform(transforms)  # Fast forward  # TODO rename

        self.activation_layer = PresetActivation(self.idx_helper)

    @property
    def is_flow_fast_forward(self):  # TODO improve, real nvp is fast forward and inverse...
        return self._fast_forward_flow

    @property
    def flow_forward_function(self):
        if self._fast_forward_flow:
            return self._forward_flow_transform.forward
        else:
            return self._forward_flow_transform.inverse

    @property
    def flow_inverse_function(self):
        if not self._fast_forward_flow:
            return self._forward_flow_transform.forward
        else:
            return self._forward_flow_transform.inverse

    def forward(self, z_K):
        # The actual transform can be forward or inverse, depending on ctor args
        v_out, _ = self.flow_forward_function(z_K)
        return self.activation_layer(v_out)




