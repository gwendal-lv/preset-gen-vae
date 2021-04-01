"""
Classes to store and transform batches of synth presets. Some functionalities are:
- can be initialized from a full preset or an inferred (incomplete) preset
- retrieve only learnable params of full presets, or fill un-learned (not inferred) values
- transform some linear parameter into categorical, and reverse this transformation
"""

from enum import Enum
from typing import Optional, Iterable, Sequence
import numpy as np

import torch
import torch.nn.functional


# Should be used instead of the str synth name to reduce loss functions computation times
class _Synth(Enum):
    GENERIC = 0  # Undefined synth - numerical params only, all are learnable
    DEXED = 1
    DIVA = 2


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
            self._param_names = ['param' for _ in range(self.full_preset_size)]
            # Default: no categorical param
            self._vst_param_learnable_model = ['num' for _ in range(self.full_preset_size)]
            self._param_cardinals = [-1 for _ in range(self.full_preset_size)]
            self._numerical_vst_params = [i for i in range(self.full_preset_size)]
            self._categorical_vst_params = []
            self._learnable_preset_size = nb_params
            self.synth_name = "generic_synth"
            self._synth = _Synth.GENERIC
        # Actual construction based on a dataset
        else:
            assert nb_params is None
            self.synth_name = dataset.synth_name
            if self.synth_name.lower() == "dexed":
                self._synth = _Synth.DEXED
            if self.synth_name.lower() == "diva":
                self._synth = _Synth.DIVA
            self._param_names = dataset.preset_param_names
            self._vst_param_learnable_model = dataset.vst_param_learnable_model
            self._param_cardinals = [dataset.get_preset_param_cardinality(vst_idx, learnable_representation=True)
                                     for vst_idx in range(dataset.total_nb_params)]
            # VSTi Param-by-param init, based on self._vst_param_learnable_model and dataset params cardinalities
            current_learnable_idx = 0
            for vst_idx in range(dataset.total_nb_params):
                if dataset.vst_param_learnable_model[vst_idx] is None:
                    self._full_to_learnable.append(None)
                elif dataset.vst_param_learnable_model[vst_idx] == 'num':
                    self._learnable_to_full.append(vst_idx)
                    self._full_to_learnable.append(current_learnable_idx)
                    current_learnable_idx += 1
                elif dataset.vst_param_learnable_model[vst_idx] == 'cat':
                    learnable_indexes = list()
                    for _ in range(self._param_cardinals[vst_idx]):
                        self._learnable_to_full.append(vst_idx)
                        learnable_indexes.append(current_learnable_idx)
                        current_learnable_idx += 1
                    self._full_to_learnable.append(learnable_indexes)
                else:
                    raise ValueError("Unknown param learning model '{}'".format(dataset.vst_param_learnable_model[vst_idx]))

            self._learnable_preset_size = current_learnable_idx
            # Final inits
            self._numerical_vst_params = dataset.numerical_vst_params
            self._categorical_vst_params = dataset.categorical_vst_params
        # Pre-compute useful indexes dicts (to use less CPU during learning). 'num' stands for 'numerical'
        # Dict keys are VST 'full-preset' indexes
        self._cat_idx_learned_as_num = dict()  # Dict of integer indexes
        self._cat_idx_learned_as_cat = dict()  # Dict of lists of integer indexes

        for vst_idx in self.categorical_vst_params:
            learnable_model = self.vst_param_learnable_model[vst_idx]
            if learnable_model is not None:
                if learnable_model == 'num':  # 1 learnable index
                    self._cat_idx_learned_as_num[vst_idx] = self.full_to_learnable[vst_idx]
                    assert isinstance(self._cat_idx_learned_as_num[vst_idx], int)
                elif learnable_model == 'cat':  # list of learnable indexes
                    self._cat_idx_learned_as_cat[vst_idx] = self.full_to_learnable[vst_idx]
                    assert isinstance(self._cat_idx_learned_as_cat[vst_idx], Iterable)
                else:
                    raise ValueError("Unknown learnable representation '{}'".format(learnable_model))
        self._num_idx_learned_as_num = dict()  # Dict of integer indexes
        self._num_idx_learned_as_cat = dict()  # Dict of lists of integer indexes
        for vst_idx in self.numerical_vst_params :
            learnable_model = self.vst_param_learnable_model[vst_idx]
            if learnable_model is not None:
                if learnable_model == 'num':  # 1 learnable index
                    self._num_idx_learned_as_num[vst_idx] = self.full_to_learnable[vst_idx]
                    assert isinstance(self._num_idx_learned_as_num[vst_idx], int)
                elif learnable_model == 'cat':  # list of learnable indexes
                    self._num_idx_learned_as_cat[vst_idx] = self.full_to_learnable[vst_idx]
                    assert isinstance(self._num_idx_learned_as_cat[vst_idx], Iterable)
                else:
                    raise ValueError("Unknown learnable representation '{}'".format(learnable_model))

    def __str__(self):
        learnable_count = sum([(0 if learn_model is None else 1) for learn_model in self._vst_param_learnable_model])
        params_str = "[PresetIndexesHelper] {} learnable VSTi parameters: ".format(learnable_count)
        for i, learn_model in enumerate(self._vst_param_learnable_model):
            if learn_model is not None:
                params_str += "    - {}.{}: {} ({})".format(i, self._param_names[i], learn_model,
                                                            self._full_to_learnable[i])
        return params_str

    @property
    def short_description(self):
        vsti_learnable_count = sum([(0 if learn_model is None else 1)
                                    for learn_model in self._vst_param_learnable_model])
        tensor_learnable_size = 0
        for learnable_indexes in self._full_to_learnable:
            if isinstance(learnable_indexes, Sequence):
                tensor_learnable_size += len(learnable_indexes)
            elif isinstance(learnable_indexes, int):
                tensor_learnable_size += 1
        return "[PresetIndexesHelper] {} learnable VSTi parameters, " \
               "learnable tensor representation size: {}".format(vsti_learnable_count, tensor_learnable_size)

    # - - - - - Properties about VSTi (full-preset) parameters - - - - -
    @property
    def full_preset_size(self):
        """ Size of a full VSTi preset (learnable and non-learnable parameters) """
        return len(self._full_to_learnable)

    @property
    def vst_param_names(self):
        return self._param_names

    @property
    def numerical_vst_params(self):
        """ VSTi-indexes of numerical synth parameters (e.g. volume, cutoff freq, ...) """
        return self._numerical_vst_params

    @property
    def categorical_vst_params(self):
        """ VSTi-indexes of categorical synth parameters (e.g. routing, LFO wave type, ...) """
        return self._categorical_vst_params

    @property
    def vst_param_learnable_model(self):
        """ None, 'num' or 'cat' (array indexes: full-preset) """
        return self._vst_param_learnable_model

    @property
    def vst_param_cardinals(self):
        return self._param_cardinals

    @property
    def full_to_learnable(self):
        """ Contains None if the param is non-learnable, an integer index if the param is learned as numerical from
        a regression, or a list of integer indexes if the param is learned as categorical.
        Array index in [0, self.full_preset_size - 1] """
        return self._full_to_learnable

    # - - - - - Pre-computed data structures, to reduce CPU usage during training - - - - -
    @property
    def cat_idx_learned_as_num(self) -> dict:
        """ Categorical VST params which are learned as numerical (default behavior).
        Dict keys are VST param indexes and values are learnable params indexes. """
        return self._cat_idx_learned_as_num

    @property
    def cat_idx_learned_as_cat(self) -> dict:
        """ Categorical VST params which are learned as categorical.
        Dict keys are VST param indexes and values are learnable params indexes. """
        return self._cat_idx_learned_as_cat

    @property
    def num_idx_learned_as_num(self) -> dict:
        """ Numerical VST params which are learned as numerical (default behavior).
        Dict keys are VST param indexes and values are learnable params indexes. """
        return self._num_idx_learned_as_num

    @property
    def num_idx_learned_as_cat(self) -> dict:
        """ Numerical VST params which are learned as categorical (to be tested).
        Dict keys are VST param indexes and values are learnable params indexes. """
        return self._num_idx_learned_as_cat

    # - - - - - Properties about learnable parameters (neural network output) - - - - -
    @property
    def learnable_preset_size(self):
        """ Size of the learnable representation of a preset. Can be smaller than self.full_preset_size
        (non-learnable params) or bigger when using categorical representations. """
        return self._learnable_preset_size

    @property
    def learnable_to_full(self):
        """ Contains the original "full-preset" (VSTi-compatible) parameter index which corresponds to
        a learnable-index index. Array Indexes in [0, self.learnable_preset_size - 1] """
        return self._learnable_to_full

    def get_numerical_learnable_indexes(self):
        """ Returns the list of indexes (learnable preset) of numerical parameters in a learnable tensor. """
        numerical_indexes = list()
        for vst_idx, learn_model in enumerate(self._vst_param_learnable_model):
            if learn_model == 'num':
                assert isinstance(self._full_to_learnable[vst_idx], int)  # extra check
                numerical_indexes.append(self._full_to_learnable[vst_idx])
        return numerical_indexes

    def get_categorical_learnable_indexes(self):
        """ Returns the list of lists of indexes (learnable preset) of categorical parameters in a learnable tensor. """
        categorical_indexes = list()
        for vst_idx, learn_model in enumerate(self._vst_param_learnable_model):
            if learn_model == 'cat':
                assert isinstance(self._full_to_learnable[vst_idx], Iterable)  # extra check
                categorical_indexes.append(self._full_to_learnable[vst_idx])
        return categorical_indexes

    def get_learnable_param_quantized_steps(self, idx):
        """ Returns None for a continuous learnable output, actual quantized steps for a discrete-numerical output,
        or [0.0, 1.0] for a categorical output neuron. """
        vst_idx = self.learnable_to_full[idx]
        learn_model = self.vst_param_learnable_model[vst_idx]
        if learn_model == 'cat':  # This learnable output is a category probability
            return np.asarray([0.0, 1.0])
        elif learn_model == 'num':  # Continuous, or discrete numerical?
            if self.vst_param_cardinals[vst_idx] >= 2:
                return np.linspace(0.0, 1.0, endpoint=True, num=self.vst_param_cardinals[vst_idx])
            else:
                return None
        else:
            raise ValueError("Unknown learnable model '{}' for idx={} (corresponding VST param idx={})"
                             .format(learn_model, idx, vst_idx))

    def get_useless_learned_params_indexes(self, preset_GT: torch.Tensor):
        """ Returns a tuple of lists of learnable indexes of useless parameters, i.e. learnable parameters which do
            not influence the output sound, and should not be used in a loss function and for backprop.
            For categorical learnable params, only the first cat-output index is returned.

            First tuple element is the list of useless numerical learned parameters.
            Second tuple element is the list of useless categorical learned parameters (first cat indexes only).

            E.g. when a Dexed Operator has a 0.0 output level, its parameters have no influence on sound and
            might have random values.

            :param preset_GT: 1D Tensor of learnable params of a given GT preset from the dataset. """
        if self._synth == _Synth.DEXED:
            # operators volumes check.
            useless_num_learn_param_indexes = []
            useless_cat_learn_param_indexes = []
            # OP switch excluded, output level excluded
            op_params_base_vst_indexes = [23, 24, 25, 26, 27, 28, 29, 30,
                                          32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
            for op_i, vst_volume_idx in enumerate([31 + 22*i for i in range(6)]):
                if isinstance(self.full_to_learnable[vst_volume_idx], int):  # numerical
                    if preset_GT[self.full_to_learnable[vst_volume_idx]].item() < 1e-3:  # OP zero-volume?
                        # add all operator-related indexes
                        cur_op_params_vst_indexes = [idx + op_i*22 for idx in op_params_base_vst_indexes]
                        for vst_idx in cur_op_params_vst_indexes:
                            learn_idx = self.full_to_learnable[vst_idx]
                            if isinstance(learn_idx, int):  # num
                                useless_num_learn_param_indexes.append(learn_idx)
                            elif isinstance(learn_idx, list):  # cat: only first cat probability output is appended
                                useless_cat_learn_param_indexes.append(learn_idx[0])
                elif self.full_to_learnable[vst_volume_idx] is None:
                    pass
                else:  # TODO If volume is categorical, we must convert....
                    raise NotImplementedError("Dexed Operator output volume learned as categorical")
            return useless_num_learn_param_indexes, useless_cat_learn_param_indexes
        else:
            return [], []


class PresetsParams:
    """
    This class basically supports two representations of presets:

    - 'full', which contains all parameters extracted from a database, with some constraints applied. Such presets
      can be used for VSTi audio rendering. Numerical representations only ().

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
            self._batch_size = self._full_presets.size(0)
        if self._learnable_presets is not None:
            assert len(self._learnable_presets.size()) == 2
            self._batch_size = self._learnable_presets.size(0)
        # Types check - float32 tensors only (some previous numpy transforms might switch to float64)
        if self._full_presets is not None:
            assert self._full_presets.dtype == self.dtype
        if self._learnable_presets is not None:
            assert self._learnable_presets.dtype == self.dtype
        # Index helpers - already built in dataset
        self.idx_helper = dataset.preset_indexes_helper  # type: PresetIndexesHelper
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
            for vst_idx, learnable_indexes in enumerate(self.idx_helper.full_to_learnable):
                # Non-learnable: default value if exists, or remains -1.0
                if self.idx_helper.vst_param_learnable_model[vst_idx] is None:
                    if vst_idx in self._default_constrained_values:  # Is key in dict?
                        full_presets[:, vst_idx] = self._default_constrained_values[vst_idx]\
                                             * torch.ones((self._learnable_presets.size(0), ))
                elif isinstance(learnable_indexes, Iterable):  # Categorical
                    n_classes = self.idx_helper.vst_param_cardinals[vst_idx]
                    classes_one_hot = self._learnable_presets[:, learnable_indexes]
                    classes = torch.argmax(classes_one_hot, dim=-1)
                    full_presets[:, vst_idx] = classes / (n_classes - 1.0)
                elif isinstance(learnable_indexes, int):  # Numerical
                    learn_idx = learnable_indexes  # type: int
                    full_presets[:, vst_idx] = self._learnable_presets[:, learn_idx]
                else:
                    raise ValueError("Bad learnable index(es) for vst idx = {}".format(vst_idx))
            return full_presets

    def get_learnable(self) -> torch.Tensor:
        if self.is_from_full_presets:
            # Pre-allocation of learnable tensor
            learnable_tensor = torch.empty((self._batch_size, self.idx_helper.learnable_preset_size),
                                           device=self._full_presets.device, requires_grad=False)
            # Numerical/categorical in VST preset are *always* stored as numerical (whatever their true
            # meaning is). So we turn only numerical to numerical/categorical
            for vst_idx, learn_indexes in enumerate(self.idx_helper.full_to_learnable):
                if learn_indexes is not None:  # Learnable params only
                    if isinstance(learn_indexes, Iterable):  # learned as categorical: one-hot encoding
                        n_classes = self.idx_helper.vst_param_cardinals[vst_idx]
                        classes = torch.round(self._full_presets[:, vst_idx] * (n_classes - 1))
                        classes = classes.type(torch.int64)  # index type required
                        # TODO check if this works with batch size > 1.... (
                        classes_one_hot = torch.nn.functional.one_hot(classes, num_classes=n_classes)
                        learnable_tensor[:, learn_indexes] = classes_one_hot.type(torch.float)
                    else:  # learned as numerical: OK, simple copy
                        idx = learn_indexes  # type: int
                        learnable_tensor[:, idx] = self._full_presets[:, vst_idx]
            return learnable_tensor
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
        # dataset must be a DexedPresetDataset
        self._algos = dataset.algos  # if all algos are used: this must remain an empty list (same as dataset's)
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
                        algo_col[row] = (self._algos[algo_dataset_index] - 1) / 31.0
            # If categorical: proper transform has not been applied
            elif self.idx_helper.vst_param_learnable_model[4] == 'cat':
                classes_one_hot = self._learnable_presets[:, self.idx_helper.full_to_learnable[4]]
                classes = torch.argmax(classes_one_hot, dim=-1)  # classes = dataset algo indexes
                # Actual algorithms must be found row-by-row...
                for row in range(classes.size(-1)):
                    # TODO manage this properly... maybe this whole class should be re-written
                    temp_algos = self._algos if len(self._algos) > 0 else list(range(1, 33))
                    full_presets[row, 4] = (temp_algos[classes[row].item()] - 1) / 31.0
        return full_presets

    def get_learnable(self) -> torch.Tensor:
        learnable_presets = super().get_learnable()
        # Algo rescale not needed if this class was built from inferred presets
        if self.is_from_full_presets:
            # TODO deactivate the "algo rescale" feature? it should be rewritten from scratch or discarded
            # Numerical algo representation is a bad idea anyways
            if self._algo_learnable_index is not None:
                """ Transforms the floating-point algorithm parameter (32 values in [0.0, 1.0]) into a new quantized
                float value (len(self.algos) values in [0.0, 1.0]). This new quantization uses the limited number
                of algorithms used in this dataset, but cannot be used for Dexed audio rendering. """
                # tensor-column - reuse for one-hot encoding
                algo_col = self._full_presets[:, 4].detach().clone()  # Vector len: batch size
                if len(self._algos) > 1:  # row-by-row quantization.... (if rescale needed)
                    for row in range(algo_col.size(0)):
                        algo_vst_index = int(round(algo_col[row].item() * 31.0))  # 32 values in [0.0, 1.0]
                        algo_dataset_index = self._algos.index(algo_vst_index + 1)  # algo numbers in [1, 32]
                        algo_col[row] = algo_dataset_index  # no new algo scale (to be used also for cat)
                if isinstance(self._algo_learnable_index, Iterable):  # categorical (one hot extracted from new scale)
                    n_classes = self.idx_helper.vst_param_cardinals[4]
                    classes = torch.round(algo_col).type(torch.int64)  # index type required
                    classes_one_hot = torch.nn.functional.one_hot(classes, num_classes=n_classes)
                    learnable_presets[:, self._algo_learnable_index] = classes_one_hot.type(torch.float)
                elif isinstance(self._algo_learnable_index, int):  # algo learn as num: rescale needed (<32 values)
                    algo_col = algo_col / (len(self._algos) - 1.0)  # New algo scale
                    learnable_presets[:, self._algo_learnable_index] = algo_col
                else:
                    raise ValueError("Unexpected vst param learnable model (expected iterable or int)")
        return learnable_presets

