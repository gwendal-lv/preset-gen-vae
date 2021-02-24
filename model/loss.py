
from collections.abc import Iterable
import numpy as np

import torch
import torch.nn as nn

from data.preset import PresetIndexesHelper

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



class SynthParamsLoss:
    """ A 'dynamic' loss which handles different representations of learnable synth parameters
    (numerical and categorical). The appropriate loss can be computed by passing a PresetIndexesHelper instance
    to this class constructor. """
    def __init__(self, idx_helper: PresetIndexesHelper, numerical_loss):
        self.idx_helper = idx_helper
        self.numerical_loss = numerical_loss
        # Pre-compute indexes lists (to use less CPU). 'num' stands for 'numerical' (not number)
        self.num_indexes = self.idx_helper.get_numerical_learnable_indexes()

    def __call__(self, u_in: torch.Tensor, u_out: torch.Tensor):
        num_loss = self.numerical_loss(u_in[:, self.num_indexes], u_out[:, self.num_indexes])
        cat_loss = 0.0  # TODO
        return num_loss + cat_loss



class QuantizedNumericalParamsLoss:
    """ 'Quantized' parameters loss: to get a meaningful (but non-differentiable) loss, inferred parameter
    values must be quantized as they would be in the synthesizer.

    Only numerical parameters are involved in this loss computation. The PresetIndexesHelper ctor argument
    allows this class to know which params are numerical.
    The loss to be applied after quantization can be passed as a ctor argument.

    This loss breaks the computation path (.backward cannot be applied to it).
    """
    def __init__(self, idx_helper: PresetIndexesHelper, numerical_loss=nn.MSELoss()):
        self.idx_helper = idx_helper
        self.numerical_loss = numerical_loss
        # Cardinality checks
        for vst_idx, _ in self.num_idx_learned_as_cat.items():
            assert self.idx_helper.vst_param_cardinals[vst_idx] > 0

    def __call__(self, u_in: torch.Tensor, u_out: torch.Tensor):
        """ Returns the loss for numerical VST params only (searched in u_in and u_out).
        Learnable representations can be numerical (in [0.0, 1.0]) or one-hot categorical.
        The type of representation has been stored in self.idx_helper """
        # Partial tensors (for final loss computation)
        minibatch_size = u_in.size(0)
        u_in_part = torch.empty((minibatch_size, 0), device=u_in.device, requires_grad=False)
        u_out_part = torch.empty((minibatch_size, 0), device=u_in.device, requires_grad=False)
        # quantize numerical learnable representations
        for vst_idx, learn_idx in self.idx_helper.num_idx_learned_as_num.items():
            param_batch = torch.unsqueeze(u_in[:, learn_idx].detach(), 1)  # Column-vector
            u_in_part = torch.cat((u_in_part, param_batch), dim=1)  # column-by-column matrix
            param_batch = torch.unsqueeze(u_out[:, learn_idx].detach(), 1)  # Column-vector
            if self.idx_helper.vst_param_cardinals[vst_idx] > 0:  # don't quantize <0 cardinal (continuous)
                cardinal = self.idx_helper.vst_param_cardinals[vst_idx]
                param_batch = torch.round(param_batch * (cardinal - 1.0)) / (cardinal - 1.0)
            u_out_part = torch.cat((u_out_part, param_batch), dim=1)  # column-by-column matrix
        # TODO convert one-hot encoded categorical learnable representations to (quantized) numerical
        for vst_idx, learn_indexes in self.idx_helper.num_idx_learned_as_cat.items():
            raise NotImplementedError("TODO cat to num conversion")
        return self.numerical_loss(u_in_part, u_out_part)



class CategoricalParamsAccuracy:
    """ Only categorical parameters are involved in this loss computation. """
    def __init__(self, idx_helper: PresetIndexesHelper, reduce=True, percentage_output=True):
        """
        :param idx_helper: allows this class to know which params are categorical
        :param reduce: If True, an averaged accuracy will be returned. If False, a dict of accuracies (keys =
          vst param indexes) is returned.
        :param percentage_output: If True, accuracies in [0.0, 100.0], else in [0.0, 1.0]
        """
        self.idx_helper = idx_helper
        self.reduce = reduce
        self.percentage_output = percentage_output

    def __call__(self, u_in: torch.Tensor, u_out: torch.Tensor):
        """ Returns accuracy (or accuracies) for all categorical VST params.
        Learnable representations can be numerical (in [0.0, 1.0]) or one-hot categorical.
        The type of representation is stored in self.idx_helper """
        accuracies = dict()
        # Accuracy of numerical learnable representations (involves quantization)
        for vst_idx, learn_idx in self.idx_helper.cat_idx_learned_as_num.items():
            cardinal = self.idx_helper.vst_param_cardinals[vst_idx]
            param_batch = torch.unsqueeze(u_in[:, learn_idx].detach(), 1)  # Column-vector
            # Class indexes, from 0 to cardinal-1
            classes_in = torch.round(param_batch * (cardinal - 1.0)).type(torch.int32)
            param_batch = torch.unsqueeze(u_out[:, learn_idx].detach(), 1)
            classes_out = torch.round(param_batch * (cardinal - 1.0)).type(torch.int32)
            accuracies[vst_idx] = (classes_in == classes_out).count_nonzero().item() / classes_in.numel()
        # TODO accuracy of one-hot encoded categorical learnable representations
        for vst_idx, learn_indexes in self.idx_helper.cat_idx_learned_as_cat:
            raise NotImplementedError("TODOOOO oooo")
            #  torch.argmax
        # Factor 100.0?
        if self.percentage_output:
            for k, v in accuracies.items():
                accuracies[k] = v * 100.0
        # Reduction if required
        if self.reduce:
            return np.asarray([v for _, v in accuracies.items()]).mean()
        else:
            return accuracies


