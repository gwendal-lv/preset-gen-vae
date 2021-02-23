"""
Implementation of the DivaDataset, based on the PresetBased abstract class.
TODO doc
"""

from .abstractbasedataset import PresetDataset
from synth import diva
import numpy as np
import pathlib
import torch
import torch.utils
import soundfile as sf

class DivaDataset(PresetDataset):
    def __init__(self, note_duration, n_fft, fft_hop,
                 midi_note=60, midi_velocity=100,  # TODO default values - try others
                 n_mel_bins=-1, mel_fmin=30.0, mel_fmax=11e3,
                 normalize_audio=False, spectrogram_min_dB=-120.0, spectrogram_normalization='min_max',
                 algos=None, operators=None,
                 restrict_to_labels=None, constant_filter_and_tune_params=True,
                 prevent_SH_LFO=False,  # TODO re-implement
                 check_constrains_consistency=True
                 ):

        super().__init__(note_duration, n_fft, fft_hop, midi_note, midi_velocity, n_mel_bins, mel_fmin, mel_fmax,
                         normalize_audio, spectrogram_min_dB, spectrogram_normalization)
        # TODO
        '''
        self.prevent_SH_LFO = prevent_SH_LFO
        assert prevent_SH_LFO is False  # TODO re-implement S&H enable/disable
        self.constant_filter_and_tune_params = constant_filter_and_tune_params
        if check_constrains_consistency:  # pre-rendered audio consistency
            self.check_audio_render_constraints_file()
        self.algos = algos if algos is not None else []
        self._operators = operators if operators is not None else [1, 2, 3, 4, 5, 6]
        self.restrict_to_labels = restrict_to_labels
        '''
        # TODO
        diva_db = diva.PresetDatabase()
        self._total_nb_presets = diva_db.get_nb_presets()
        self._total_nb_params = diva_db.get_nb_params()
        self._param_names = diva_db.get_param_names()

        '''
        self.learnable_params_idx = list(range(0, self._total_nb_params))
        if self.constant_filter_and_tune_params:  # TODO Comment connaitre mes params non-learnable
            for idx in [0]:
                self.learnable_params_idx.remove(idx)
        for i_op in range(6):
            if not (i_op+1) in self._operators:  # If disabled: we remove all corresponding learnable params
                for idx in range(21):  # Don't remove the 22nd param (OP on/off selector) yet
                    self.learnable_params_idx.remove(23 + 22*i_op + idx)  # idx 23 is the first param of op 1
        for col in [44, 66, 88, 110, 132, 154]:
            self.learnable_params_idx.remove(col)
        '''
        # TODO Fin

        # - - - Valid presets - UIDs of presets, and not their database row index - - -
        # Select valid presets by algorithm
        # TODO Format perso différent ? Changer ?
        '''
        if len(self.algos) == 0:  # All presets are valid
            self.valid_preset_UIDs = diva_db.all_presets["index_preset"].values
        else:
            if len(self.algos) == 1:
                self.learnable_params_idx.remove(4)  # Algo parameter column idx
            valid_presets_row_indexes = diva_db.get_preset_indexes_for_algorithms(self.algos)
            self.valid_preset_UIDs = diva_db.all_presets\
                .iloc[valid_presets_row_indexes]['index_preset'].values
        # Select valid presets by label. We build a list of list-indexes to remove
        if self.restrict_to_labels is not None:
            self.valid_preset_UIDs = [uid for uid in self.valid_preset_UIDs
                                      if any([self.is_label_included(l) for l in self.get_labels_name(uid)])]
        # - - - DB class deleted (we need a low memory usage for multi-process dataloaders) - - -
        '''
        del diva_db
        # - - - Parameters constraints, cardinality, indexes management, ... - - -
        # Param cardinalities are stored - Dexed cardinality involves a short search which can be avoided
        # This cardinality is the LEARNING REPRESENTATION cardinality - will be used for categorical representations
        '''
        self._params_cardinality = np.asarray([diva.Diva.get_param_cardinality(idx)
                                               for idx in range(self.total_nb_params)])
        self._params_default_values = dict()
        # Algo cardinality is manually set. We consider an algo-limited DX7 to be a new synth
        if len(self.algos) > 0:  # len 0 means all algorithms are used
            self._params_cardinality[4] = len(self.algos)
        if len(self.algos) == 1:  # 1 algo: constrained constant param
            self._params_default_values[4] = (self.algos[0] - 1) / 31.0
        # cardinality 1 for constrained parameters (operators are always constrained)
        self._params_cardinality[[44, 66, 88, 110, 132, 154]] = np.ones((6,), dtype=np.int)
        for op_i, op_switch_idx in enumerate([44, 66, 88, 110, 132, 154]):
            self._params_default_values[op_switch_idx] = 1.0 if ((op_i+1) in self._operators) else 0.0
        if self.constant_filter_and_tune_params:
            # TODO DEFAULT
            self._params_cardinality[[0, 1, 2, 3, 13]] = np.ones((5,), dtype=np.int)
            self._params_default_values[0] = 1.0 # OUTPUT
        # None / Numerical / Categorical learnable status array
        self._vst_param_learnable_model = list()
        for idx in range(self.total_nb_params):
            if idx not in self.learnable_params_idx:
                self._vst_param_learnable_model.append(None)
            else:
                # TODO categorical representation for some preset params
                self._vst_param_learnable_model.append('num')  # Default: numerical
        # - - - Final initializations - - -
        self._preset_idx_helper = PresetIndexesHelper(self)
        '''
        self._load_spectrogram_stats()  # Must be called after super() ctor

    @property
    def synth_name(self):
        return "Diva"

    @property
    def total_nb_presets(self):
        return self._total_nb_presets

    @property
    def vst_param_learnable_model(self):
        return self._vst_param_learnable_model

    @property
    def params_default_values(self):
        return self._params_default_values

    @property
    def total_nb_params(self):
        return self._total_nb_params

    @property
    def preset_indexes_helper(self):
        return self._preset_idx_helper

    @property
    def preset_param_names(self):
        return self._param_names

if __name__ == "__main__":
    # TODO regenerate audio and spectrogram stats files
    # ============== DATA RE-GENERATION - FROM config.py ==================
    regenerate_wav = False  # quite long (15min, full dataset, 1 midi note)
    regenerate_spectrograms_stats = False  # approx 3 min

    import sys

    sys.path.append(pathlib.Path(__file__).parent.parent)
    import config  # Dirty path trick to import config.py from project root dir

    # operators = config.model.dataset_synth_args[1]  # Custom operators limitation?
    operators = [1, 2, 3, 4, 5, 6]

    # No label restriction, no normalization, etc...
    # But: OPERATORS LIMITATIONS and DEFAULT PARAM CONSTRAINTS (main params (filter, transpose,...) are constant)
    diva_dataset = DivaDataset(note_duration=config.model.note_duration,
                                 n_fft=config.model.stft_args[0], fft_hop=config.model.stft_args[1],
                                 n_mel_bins=config.model.mel_bins,
                                 spectrogram_normalization=None,  # No normalization: we want to compute stats
                                 algos=None,  # allow all algorithms
                                 restrict_to_labels=None,
                                 operators=operators,  # Operators limitation (config.py)
                                 spectrogram_min_dB=config.model.spectrogram_min_dB,
                                 check_constrains_consistency=False)
    for i in range(0):
        test = diva_dataset[i]  # try get an item - for debug purposes
        print(diva_dataset)  # All files must be pre-rendered before printing
        print(test)

    if regenerate_wav:
        # WRITE ALL WAV FILES (approx. 10.5Go for 4.0s audio, 1 midi note)
        diva_dataset.generate_wav_files()
    if regenerate_spectrograms_stats:
        # whole-dataset stats (for proper normalization)
        diva_dataset.compute_and_store_spectrograms_stats()
    # ============== DATA RE-GENERATION - FROM config.py ==================

