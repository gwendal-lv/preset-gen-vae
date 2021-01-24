import os
import torch
import torch.nn as nn
import torch.utils
import torchaudio.transforms
import numpy as np
import copy
import sys

from synth import dexed

# See https://github.com/pytorch/audio/issues/903
#torchaudio.set_audio_backend("sox_io")


class DexedDataset(torch.utils.data.Dataset):
    def __init__(self, algos=None, constant_filter_and_tune_params=True, prevent_SH_LFO=True,
                 midi_note=60, midi_velocity=100,  # TODO default values - try others
                 n_fft=1024, fft_hop=512,  # obtained spectrogram is roughly the same size as 22.05kHz audio
                 n_mel_bins=-1, mel_fmin=30.0, mel_fmax=11e3):
        """
        Allows access to Dexed preset values and names, and generates spectrograms and corresponding
        parameters values. Can manage a reduced number of synth parameters (using default values for non-
        learnable params).

        :param algos: List. Can be used to limit the DX7 algorithms included in this dataset. Set to None
        to use all available algorithms
        :param constant_filter_and_tune_params: if True, the main filter and the main tune settings are default
        :param prevent_SH_LFO: if True, replaces the SH random LFO by a square-wave LFO
        :param n_mel_bins: Number of frequency bins for the Mel-spectrogram. If -1, the normal STFT will be
        used instead.
        """
        self.mel_fmax = mel_fmax
        self.mel_fmin = mel_fmin
        self.n_mel_bins = n_mel_bins
        self.fft_hop = fft_hop
        self.n_fft = n_fft
        # Default spectr. params: Hann window, power spectrogram, un-normalized
        self.torch_spectrogram = nn.Sequential(
            torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=fft_hop,
                                              pad=0),  # conv kernels should zero-pad
            torchaudio.transforms.AmplitudeToDB('power', top_db=80)
        )
        self.midi_note = midi_note
        self.midi_velocity = midi_velocity
        self.prevent_SH_LFO = prevent_SH_LFO
        self.constant_filter_and_tune_params = constant_filter_and_tune_params
        self.algos = algos if algos is not None else []
        # Full DB read and stored into a df and a np.array
        self.dexed_db = dexed.PresetDatabase()
        self.learnable_params_idx = list(range(0, self.dexed_db.presets_mat.shape[1]))
        # - - - Pre-processing of parameters (for all algorithms)
        if self.constant_filter_and_tune_params:  # (see dexed db exploration notebook)
            self.dexed_db.presets_mat[:, 0] = 1.0  # filter cutoff
            self.dexed_db.presets_mat[:, 1] = 0.0  # filter reso
            self.dexed_db.presets_mat[:, 2] = 1.0  # filter vol
            self.dexed_db.presets_mat[:, 3] = 0.5  # master tune
            self.dexed_db.presets_mat[:, 13] = 0.5  # middle-C to C3
            for idx in [0, 1, 2, 3, 13]:
                self.learnable_params_idx.remove(idx)
        if self.prevent_SH_LFO:
            for row in range(self.dexed_db.presets_mat.shape[0]):
                if self.dexed_db.presets_mat[row, 12] > 0.95:  # S&H wave corresponds to a 1.0 param value
                    self.dexed_db.presets_mat[row, 12] = 4.0 / 5.0  # Square wave is number 4/6
        # All oscillators are always ON (see dexed db exploration notebook)
        for col in [44, 66, 88, 110, 132, 154]:
            self.dexed_db.presets_mat[:, col] = 1.0  # OPx switch ON
            self.learnable_params_idx.remove(col)
        # - - - Valid presets - row indexes of the main preset values matrix
        if len(self.algos) == 0:  # All presets are valid
            self.valid_presets_idx = np.arange(0, self.dexed_db.presets_mat.shape[0])
        else:
            if len(self.algos) == 1:
                self.learnable_params_idx.remove(4)  # Algo parameter column idx
            self.valid_presets_idx = list()
            for algo in self.algos:
                self.valid_presets_idx += self.dexed_db.get_preset_indexes_for_algorithm(algo)

    def __len__(self):
        return len(self.valid_presets_idx)

    def __getitem__(self, i):
        """ Returns a tuple containing a 2D tensor spectrogram (1st dim: time; 2nd dim: frequency),
        a 1D tensor of parameter values, and a 1d tensor with the midi note and velocity """
        midi_note = self.midi_note
        midi_velocity = self.midi_velocity
        # on-the-fly audio rendering (easier framework for future data augmentation tests)
        dexed_renderer = dexed.Dexed()  # reloads the VST to prevent hanging notes/sounds
        preset = self.dexed_db.get_params_in_plugin_format(self.dexed_db.presets_mat[i, :])
        dexed_renderer.assign_preset(preset)
        if self.constant_filter_and_tune_params:
            dexed_renderer.set_default_general_filter_and_tune_params()
        if self.prevent_SH_LFO:
            dexed_renderer.prevent_SH_LFO()
        x_wav = dexed_renderer.render_note(midi_note, midi_velocity)
        # Tuple output. Warning: torch.from_numpy does not copy values
        return self.torch_spectrogram(torch.tensor(x_wav, dtype=torch.float32)), \
               torch.tensor(self.dexed_db.presets_mat[i, self.learnable_params_idx], dtype=torch.float32), \
               torch.tensor([midi_note, midi_velocity], dtype=torch.int8)

    def get_spectrogram_tensor_size(self):
        """ Returns the size of the first tensor (2D image) returned by this dataset. """
        dummy_spectrogram, _, _ = self.__getitem__(0)
        return dummy_spectrogram.size()

    def get_param_tensor_size(self):
        """ Returns the length of the second tensor returned by this dataset. """
        return len(self.learnable_params_idx)

    def __str__(self):
        return "Dataset of {} Dexed presets. {} learnable synth params, {} fixed params.\n{}" \
            .format(len(self), len(self.learnable_params_idx),
                    self.dexed_db.presets_mat.shape[1] - len(self.learnable_params_idx),
                    self.dexed_db.get_size_info())


if __name__ == "__main__":
    dexed_dataset = DexedDataset()
    print(dexed_dataset)

    one_sample = dexed_dataset[0]
    print(one_sample)

    # TODO stats sur les spectrogrammes de tous le dataset - sauvegarder en CSV

    # Dataloader test
    if False:
        dexed_dataloader = torch.utils.data.DataLoader(dexed_dataset, batch_size=2, shuffle=False,
                                                       num_workers=os.cpu_count() * 9 // 10, persistent_workers=True,
                                                       pin_memory=True)
        for batch_idx, sample in enumerate(dexed_dataloader):
            print(batch_idx)
            print(sample)
