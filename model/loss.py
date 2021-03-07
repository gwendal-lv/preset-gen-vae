
from collections.abc import Iterable
import numpy as np

import torch
import torch.nn as nn

from data.preset import PresetIndexesHelper


class L2Loss:
    """
    L2 (squared difference) loss, with customizable normalization options.

    When used to model the reconstruction probability p_theta( x | zK ), normalization has strong
    implications on the p_theta( x | zK ) model itself.
    E.g., for a 1-element batch, the non-normalized L2 loss implies a learned mean, fixed 1/√2 std
    gaussian model for each element of x.

    By normalizing the L2 error (i.e. MSE error), the fixed std is multiplied by √(nb of elements of x)
    (e.g. approx *300 for a 250x350 pixels spectrogram)

    Normalization over batch dimension should always be performed (monte-carlo log-proba estimation).
    """
    pass  # TODO


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
    to this class constructor.

    The categorical loss is categorical cross-entropy. """
    def __init__(self, idx_helper: PresetIndexesHelper, numerical_loss, categorical_loss_factor=0.2,
                 prevent_useless_params_loss=True):
        """

        :param idx_helper: PresetIndexesHelper instance, created by a PresetDatabase, to convert vst<->learnable params
        :param numerical_loss: Loss class instance to compute loss on numerical-represented learnable parameters.
            The given loss function should perform a batch mean reduction.
        :param categorical_loss_factor: Factor to be applied to the categorical cross-entropy loss, which is
            much greater than the 'corresponding' MSE loss (if the parameter was learned as numerical)
        :param prevent_useless_params_loss: If True, the class will search for useless params (e.g. params which
            correspond to a disabled oscillator and have no influence on the output sound). This introduces a
            TODO describe overhead here
        """
        self.idx_helper = idx_helper
        self.numerical_loss = numerical_loss
        self.cat_loss_factor = categorical_loss_factor
        self.prevent_useless_params_loss = prevent_useless_params_loss
        # Pre-compute indexes lists (to use less CPU). 'num' stands for 'numerical' (not number)
        self.num_indexes = self.idx_helper.get_numerical_learnable_indexes()
        self.cat_indexes = self.idx_helper.get_categorical_learnable_indexes()

    def __call__(self, u_in: torch.Tensor, u_out: torch.Tensor):
        """ Categorical parameters must be one-hot encoded. """
        # At first: we search for useless parameters (whose loss should not be back-propagated)
        useless_num_learn_param_indexes, useless_cat_learn_param_indexes = list(), list()
        batch_size = u_in.shape[0]
        if self.prevent_useless_params_loss:
            for row in range(batch_size):
                num_indexes, cat_indexes = self.idx_helper.get_useless_learned_params_indexes(u_in[row, :])
                useless_num_learn_param_indexes.append(num_indexes)
                useless_cat_learn_param_indexes.append(cat_indexes)
        num_loss = 0.0  # - - - numerical loss - - -
        if len(self.num_indexes) > 0:
            if self.prevent_useless_params_loss:
                # TODO apply a 0.0 factor for disabled parameters (e.g. Dexed operator w/ output level 0.0)
                for row in range(u_in.shape[0]):
                    for num_idx in self.num_indexes:
                        if num_idx in useless_num_learn_param_indexes[row]:
                            u_in[row, num_idx] = 0.0
                            u_out[row, num_idx] = 0.0
            num_loss = self.numerical_loss(u_in[:, self.num_indexes], u_out[:, self.num_indexes])
        cat_loss = 0.0  # - - - categorical loss - - -
        if len(self.cat_indexes) > 0:
            # For each categorical output (separate loss computations...)
            for cat_learn_indexes in self.cat_indexes:  # type: list
                # TODO don't compute cat loss for disabled parameters (e.g. Dexed operator w/ output level 0.0)
                rows_to_remove = list()
                if self.prevent_useless_params_loss:
                    for row in range(batch_size):  # Need to check cat index 0 only
                        if cat_learn_indexes[0] in useless_cat_learn_param_indexes[row]:
                            rows_to_remove.append(row)
                useful_rows = None  # None means that the full batch is useful
                if len(rows_to_remove) > 0:  # If this batch contains useless inferred params
                    useful_rows = list(range(0, batch_size))
                    for row in rows_to_remove:
                        useful_rows.remove(row)
                # Direct cross-entropy computation. The one-hot target is used to select only q output probabilities
                # corresponding to target classes with p=1. We only need a limited number of output probabilities
                # (they actually all depend on each other thanks to the softmax output layer).
                target_one_hot = u_in[:, cat_learn_indexes].bool()  # Will be used for tensor-element selection
                if useful_rows is not None:  # Some rows can be discarded from loss computation
                    target_one_hot = target_one_hot[useful_rows, :]
                # Then the cross-entropy can be computed (simplified formula thanks to p=1.0 one-hot odds)
                q_odds = u_out[:, cat_learn_indexes]  # contains all q odds
                if useful_rows is not None:  # Some rows can be discarded from loss computation
                    q_odds = q_odds[useful_rows, :]
                q_odds = q_odds[target_one_hot]
                # normalization vs. batch size
                cat_loss += - torch.sum(torch.log(q_odds)) / (batch_size - len(rows_to_remove))
            cat_loss = cat_loss / len(self.cat_indexes)  # Normalization vs. number of categorical-learned params
        # losses weighting - Cross-Entropy is usually be much bigger than MSE. num_loss
        return num_loss + cat_loss * self.cat_loss_factor



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
        for vst_idx, _ in self.idx_helper.num_idx_learned_as_cat.items():
            assert self.idx_helper.vst_param_cardinals[vst_idx] > 0
        # Number of numerical parameters considered for this loss (after cat->num conversions). For tensor pre-alloc
        self.num_params_count = len(self.idx_helper.num_idx_learned_as_num)\
                                + len(self.idx_helper.num_idx_learned_as_cat)

    def __call__(self, u_in: torch.Tensor, u_out: torch.Tensor):
        """ Returns the loss for numerical VST params only (searched in u_in and u_out).
        Learnable representations can be numerical (in [0.0, 1.0]) or one-hot categorical.
        The type of representation has been stored in self.idx_helper """
        # Partial tensors (for final loss computation)
        minibatch_size = u_in.size(0)
        # pre-allocate tensors
        u_in_num = torch.empty((minibatch_size, self.num_params_count), device=u_in.device, requires_grad=False)
        u_out_num = torch.empty((minibatch_size, self.num_params_count), device=u_in.device, requires_grad=False)
        cur_num_tensors_col = 0  # Column-by-column tensors filling
        # quantize numerical learnable representations
        for vst_idx, learn_idx in self.idx_helper.num_idx_learned_as_num.items():
            param_batch = u_in[:, learn_idx].detach()
            u_in_num[:, cur_num_tensors_col] = param_batch  # Data copy - does not modify u_in
            param_batch = u_out[:, learn_idx].detach()
            if self.idx_helper.vst_param_cardinals[vst_idx] > 0:  # don't quantize <0 cardinal (continuous)
                cardinal = self.idx_helper.vst_param_cardinals[vst_idx]
                param_batch = torch.round(param_batch * (cardinal - 1.0)) / (cardinal - 1.0)
            u_out_num[:, cur_num_tensors_col] = param_batch
            cur_num_tensors_col += 1
        # convert one-hot encoded categorical learnable representations to (quantized) numerical
        for vst_idx, learn_indexes in self.idx_helper.num_idx_learned_as_cat.items():
            cardinal = len(learn_indexes)
            # Classes as column-vectors (for concatenation)
            in_classes = torch.argmax(u_in[:, learn_indexes], dim=-1).detach().type(torch.float)
            u_in_num[:, cur_num_tensors_col] = in_classes / (cardinal-1.0)
            out_classes = torch.argmax(u_out[:, learn_indexes], dim=-1).detach().type(torch.float)
            u_out_num[:, cur_num_tensors_col] = out_classes / (cardinal-1.0)
            cur_num_tensors_col += 1
        assert cur_num_tensors_col == self.num_params_count  # size check
        return self.numerical_loss(u_out_num, u_in_num)  # Positive diff. if output > input



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
            target_classes = torch.round(param_batch * (cardinal - 1.0)).type(torch.int32)  # New tensor allocated
            param_batch = torch.unsqueeze(u_out[:, learn_idx].detach(), 1)
            out_classes = torch.round(param_batch * (cardinal - 1.0)).type(torch.int32)  # New tensor allocated
            accuracies[vst_idx] = (target_classes == out_classes).count_nonzero().item() / target_classes.numel()
        # accuracy of one-hot encoded categorical learnable representations
        for vst_idx, learn_indexes in self.idx_helper.cat_idx_learned_as_cat.items():
            target_classes = torch.argmax(u_in[:, learn_indexes], dim=-1)  # New tensor allocated
            out_classes = torch.argmax(u_out[:, learn_indexes], dim=-1)  # New tensor allocated
            accuracies[vst_idx] = (target_classes == out_classes).count_nonzero().item() / target_classes.numel()
        # Factor 100.0?
        if self.percentage_output:
            for k, v in accuracies.items():
                accuracies[k] = v * 100.0
        # Reduction if required
        if self.reduce:
            return np.asarray([v for _, v in accuracies.items()]).mean()
        else:
            return accuracies


