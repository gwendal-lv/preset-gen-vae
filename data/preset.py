"""
Classes to store and transform batches of synth presets. Some functionalities are:
- can be initialized from a full preset or an inferred (incomplete) preset
- retrieve only learnable params of full presets, or fill un-learned (not inferred) values
- transform some linear parameter into categorical, and reverse this transformation
"""

from typing import Optional
from collections.abc import Iterable
import numpy as np

import torch



class PresetIndexesHelper:
    """ Class to help convert a full-preset parameter index (VSTi-compatible) into its corresponding learned
    parameter index (or indexes, if the parameter is learned as categorical), and help reverse this conversion.

    Can be constructed once from a PresetDataset, then passed as argument to PresetsParams instances.

    This is required for converting presets (see PresetsParams) between 'full-VSTi' and 'learnable' representations,
    and to compute per-parameter losses (categorical vs. MSE loss). """

    def __init__(self, dataset=None, nb_params=None):
        """ Builds an indexes translator given a dataset.

        For convenience: an identity translator can be built given the number of params (with dataset == None). """
        self._full_to_learnable = list()
        self._learnable_to_full = list()
        # Identity default translator
        if dataset is None:
            assert nb_params is not None
            self._full_to_learnable = np.arange(0, nb_params)
            self._learnable_to_full = self._full_to_learnable
            self._vst_param_learnable_model = ['num' for _ in range(self.full_preset_size)]
            self._param_names = ['param' for _ in range(self.full_preset_size)]
        # Actual construction based on a dataset
        else:
            assert nb_params is None
            self._param_names = dataset.preset_param_names
            self._vst_param_learnable_model = dataset.vst_param_learnable_model
            # VSTi Param-by-param init, based on self._vst_param_learnable_model and dataset params cardinalities
            current_learnable_idx = 0
            for i in range(dataset.total_nb_params):
                if dataset.vst_param_learnable_model[i] is None:
                    self._full_to_learnable.append(None)
                elif dataset.vst_param_learnable_model[i] == 'num':
                    self._full_to_learnable.append(current_learnable_idx)
                    self._learnable_to_full.append(i)
                    current_learnable_idx += 1
                elif dataset.vst_param_learnable_model[i] == 'cat':
                    raise NotImplementedError("TODOOOOO categorical indexes")
                else:
                    raise ValueError("Unknown param learning model '{}'".format(dataset.vst_param_learnable_model[i]))

    def __str__(self):
        learnable_count = sum([(0 if learn_model is None else 1) for learn_model in self._vst_param_learnable_model])
        params_str = "[PresetIndexesHelper] {} learnable parameters: ".format(learnable_count)
        for i, learn_model in enumerate(self._vst_param_learnable_model):
            if learn_model is not None:
                params_str += "    - {}.{}: {} ({})".format(i, self._param_names[i], learn_model,
                                                      self._full_to_learnable[i])
        return params_str

    @property
    def vst_param_learnable_model(self):
        """ None, 'num' or 'cat' (array indexes: full-preset) """
        return self._vst_param_learnable_model

    @property
    def full_preset_size(self):
        """ Size of a full VSTi preset (learnable and non-learnable parameters) """
        return len(self._full_to_learnable)

    @property
    def learnable_preset_size(self):
        """ Size of the learnable representation of a preset. Can be smaller than self.full_preset_size
        (non-learnable params) or bigger when using categorical representations. """
        return len(self._learnable_to_full)

    @property
    def full_to_learnable(self):
        """ Contains None if the param is non-learnable, an integer index if the param is learned as numerical from
        a regression, or a list of integer indexes if the param is learned as categorical.
        Array index in [0, self.full_preset_size - 1] """
        return self._full_to_learnable

    @property
    def learnable_to_full(self):
        """ Contains the original "full-preset" (VSTi-compatible) parameter index which corresponds to
        a learnable-index index. Array Indexes in [0, self.learnable_preset_size - 1] """
        return self._learnable_to_full



class PresetsParams:
    """
    This class basically supports two representations of presets:

    - 'full', which contains all parameters extracted from a database, with some constraints applied. Such presets
      can be used for VSTi audio rendering.

    - 'learnable', which contains only the learnable parameters, with transformations on some learnable
      params (e.g. linear to categorical, distortion on linear values, ...)
    """

    def __init__(self, dataset,
                 full_presets: Optional[torch.Tensor] = None, learnable_presets: Optional[torch.Tensor] = None):
        """
        Inits from a batch of presets (a single preset must be unsqueezed to be passed to this constructor).

        :param full_presets: Full presets as read from the database (no constraints applied)
        :param learnable_presets: Learnable or inferred presets (with distortion, linear to categorical transforms, ...)
        :param dataset: abstractbasedataset.PresetDataset instance
        """
        # Construction for either a full or learnable batch of presets (not both)
        assert (full_presets is None) != (learnable_presets is None)
        self._is_from_full_preset = full_presets is not None
        self._full_presets = full_presets
        self._learnable_presets = learnable_presets
        # attributes from dataset
        self._learnable_params_idx = dataset.learnable_params_idx
        self._default_constrained_values = dataset.params_default_values
        self._params_cardinality = [dataset.get_preset_param_cardinality(idx)
                                    for idx in range(dataset.total_nb_params)]
        # Size checks - 2D Tensors only
        if self._full_presets is not None:
            assert len(self._full_presets.size()) == 2
        if self._learnable_presets is not None:
            assert len(self._learnable_presets.size()) == 2
        # Types check - float32 tensors only (some previous numpy transforms might switch to float64)
        if self._full_presets is not None:
            assert self._full_presets.dtype == self.dtype
        if self._learnable_presets is not None:
            assert self._learnable_presets.dtype == self.dtype
        # Index helpers - already built in dataset
        self.idx_helper = dataset.preset_indexes_helper

        # TODO data structures to ease categorical<->linear transformations
        # TODO change learnable indexes if categorical outputs

    @property
    def dtype(self):
        return torch.float32

    @property
    def is_from_full_presets(self):
        """ Returns True if this class was constructed from a batch of full presets, else False. """
        return self._is_from_full_preset

    def get_full(self, apply_constraints=True) -> torch.Tensor:
        if self.is_from_full_presets:
            if not apply_constraints:
                return self._full_presets
            else:  # apply constrains - copied data to prevent modification of the original presets
                constrained_full_preset = self._full_presets.clone().detach()
                for k, v in self._default_constrained_values.items():
                    constrained_full_preset[:, k] = v * torch.ones((constrained_full_preset.size(0), ))
                return constrained_full_preset
        else:  # From learnable presets
            full_presets = -0.1 * torch.ones((self._learnable_presets.size(0), self.idx_helper.full_preset_size))
            # We use the index translator to perform a param-by-param fill
            for i in range(self.idx_helper.full_preset_size):
                # Non-learnable: default value if exists, or remains -1.0
                if self.idx_helper.vst_param_learnable_model[i] is None:
                    if i in self._default_constrained_values:  # Is key in dict?
                        full_presets[:, i] = self._default_constrained_values[i]\
                                             * torch.ones((self._learnable_presets.size(0), ))
                elif self.idx_helper.vst_param_learnable_model[i] == 'num':  # Numerical
                    full_presets[:, i] = self._learnable_presets[:, self.idx_helper.full_to_learnable[i]]
                else:  # TODO turn categorical into numerical
                    raise NotImplementedError('TODO')
            return full_presets

    def get_learnable(self) -> torch.Tensor:
        if self.is_from_full_presets:
            # Copy non-constrained columns, to prevent modification of the original presets
            learnable_presets = torch.Tensor(self._full_presets[:, self._learnable_params_idx])
            # TODO numerical to categorical
            return learnable_presets
        else:
            return self._learnable_presets

    # TODO learnable representation, numerical-only (no category)

    # TODO quantize learned representation



class DexedPresetsParams(PresetsParams):
    """ A PresetsParams class with DX7-specific parameter constraints (e.g. rescaled algorithm parameter,
    S&H LFO, ...). Can be build using individual arguments, or using a DexedDataset argument. """
    def __init__(self, dataset,
                 full_presets: Optional[torch.Tensor] = None, learnable_presets: Optional[torch.Tensor] = None):
        super().__init__(dataset, full_presets, learnable_presets)
        self._algos = dataset.algos  # dataset must be a DexedPresetDataset
        # find algo column index in a learnable presets params tensor
        self._algo_learnable_index = self.idx_helper.full_to_learnable[4]

    def get_full(self, apply_constraints=True) -> torch.Tensor:
        full_presets = super().get_full(apply_constraints)
        if not self.is_from_full_presets:
            if self.idx_helper.vst_param_learnable_model[4] == 'num':  # algo learnable: rescale needed (to 32 values)
                # Direct tensor-column modification
                algo_col = full_presets[:, 4]  # Vector (len = batch size)
                if len(self._algos) > 1:  # row-by-row quantization.... (if rescale needed)
                    for row in range(algo_col.size(0)):
                        algo_dataset_index = int(round(algo_col[row].item() * (len(self._algos) - 1.0)))
                        print("algo de-scaling. algo_col={} --> learnable index = {}"
                              .format(algo_col, algo_dataset_index))
                        algo_col[row] = (self._algos[algo_dataset_index] - 1) / 31.0
            # If categorical: proper transform has already been applied by super().get_full()? TODO check card issue
        return full_presets

    def get_learnable(self) -> torch.Tensor:
        learnable_presets = super().get_learnable()
        # Algo rescale not needed if this class was built from inferred presets
        if self.is_from_full_presets:
            """ Transforms the floating-point algorithm parameter (32 values in [0.0, 1.0]) into a new quantized
            float value (len(self.algos) values in [0.0, 1.0]). This new quantization uses the limited number
            of algorithms used in this dataset, but cannot be used for Dexed audio rendering. """
            if self.idx_helper.vst_param_learnable_model[4] == 'num':  # algo learnable: rescale needed (<32 values)
                # Direct tensor-column modification
                algo_col = learnable_presets[:, self._algo_learnable_index]  # Vector (len = batch size)
                if len(self._algos) > 1:  # row-by-row quantization.... (if rescale needed)
                    for row in range(algo_col.size(0)):
                        algo_vst_index = int(round(algo_col[row].item() * 31.0))  # 32 values in [0.0, 1.0]
                        algo_dataset_index = self._algos.index(algo_vst_index + 1)  # algo numbers in [1, 32]
                        algo_col[row] = algo_dataset_index / (len(self._algos) - 1.0)  # New algo scale
            # If categorical: proper transform has already been applied by super().get_full()? TODO check card issue
        return learnable_presets

