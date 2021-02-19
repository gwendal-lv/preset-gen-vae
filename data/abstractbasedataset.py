""" TODO proper doc """

import os
import pathlib
from abc import ABC, abstractmethod  # Abstract Base Class
import pandas as pd
import json
import torch
import torch.utils
import numpy as np

import utils.audio

# See https://github.com/pytorch/audio/issues/903
#torchaudio.set_audio_backend("sox_io")


class PresetDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, note_duration,
                 n_fft, fft_hop,  # ftt 1024 hop=512: spectrogram is approx. the size of 5.0s@22.05kHz audio
                 midi_note=60, midi_velocity=100,  # TODO default values - try others
                 n_mel_bins=-1, mel_fmin=30.0, mel_fmax=11e3,
                 normalize_audio=False, spectrogram_min_dB=-120.0, spectrogram_normalization='min_max'
                 ):
        """
        Abstract Base Class for any synthesizer presets dataset.

        :param note_duration: Tuple: MIDI Note (on_duration, off_duration) in seconds
        :param n_fft: Width of the FFT window for spectrogram computation
        :param fft_hop: STFT hop length (in samples)
        :param midi_note:  Default dataset MIDI note value (used to pre-render and to load .wav file)
        :param midi_velocity:  Default dataset MIDI velocity value (used to pre-render and to load .wav file)
        :param n_mel_bins: Number of frequency bins for the Mel-spectrogram. If -1, the normal STFT will be used
        :param mel_fmin: TODO implement
        :param mel_fmax: TODO implement
        :param normalize_audio:  If True, audio from RenderMan will be normalized
        :param spectrogram_min_dB:  Noise-floor threshold value for log-scale spectrograms
        :param spectrogram_normalization: 'min_max' to get output spectrogram values in [-1, 1], or 'mean_std'
            to get zero-mean unit-variance output spectrograms. None to disable normalization.
        """
        self.note_duration = note_duration
        self.n_fft = n_fft
        self.fft_hop = fft_hop
        self.midi_note = midi_note
        self.midi_velocity = midi_velocity
        self.n_mel_bins = n_mel_bins
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.normalize_audio = normalize_audio
        # - - - - - Attributes to be set by the child concrete class - - - - -
        self.valid_preset_UIDs = np.zeros((0,))  # UIDs (may be indexes) of valid presets for this dataset
        self.learnable_params_idx = list()  # Indexes of learnable VSTi params (some params may be constant or unused)
        # - - - Spectrogram utility class - - -
        if self.n_mel_bins <= 0:
            self.spectrogram = utils.audio.Spectrogram(self.n_fft, self.fft_hop, spectrogram_min_dB)
        else:  # TODO do not hardcode Fs?
            self.spectrogram = utils.audio.MelSpectrogram(self.n_fft, self.fft_hop, spectrogram_min_dB,
                                                          self.n_mel_bins, 22050)
        # spectrogram min/max/mean/std statistics: must be loaded after super() ctor (depend on child class args)
        self.spectrogram_normalization = spectrogram_normalization
        self.spec_stats = None

    @property
    @abstractmethod
    def synth_name(self):
        pass

    def __str__(self):
        return "Dataset of {}/{} {} presets. {} learnable synth params, {} fixed params. " \
               "{} Spectrogram items, size={}, min={:.1f}dB, normalization:{}" \
            .format(len(self), self.total_nb_presets, self.synth_name, len(self.learnable_params_idx),
                    self.total_nb_params - len(self.learnable_params_idx),
                    ("Linear" if self.n_mel_bins <= 0 else "Mel"), self.get_spectrogram_tensor_size(),
                    self.spectrogram.min_dB, self.spectrogram_normalization)

    def __len__(self):  # Required for any torch.utils.data.Dataset
        return len(self.valid_preset_UIDs)

    def __getitem__(self, i):
        """ Returns a tuple containing a 2D scaled dB spectrogram tensor (1st dim: freq; 2nd dim: time),
        a 1D tensor of parameter values in [0;1], and a 1d tensor with remaining int info (preset UID, midi note, vel).

        If this dataset generates audio directly from the synth, only 1 dataloader is allowed.
        A 30000 presets dataset require approx. 7 minutes to be generated on 1 CPU. """
        # TODO on-the-fly audio generation. We should try:
        #  - Use shell command to run a dedicated script. The script writes AUDIO_SAMPLE_TEMP_ID.wav
        #  - wait for the file to be generated on disk (or for the command to notify... something)
        #  - read and delete this .wav file
        midi_note = self.midi_note
        midi_velocity = self.midi_velocity
        # loading params and wav file TODO load labels
        preset_UID = self.valid_preset_UIDs[i]
        preset_params = self.get_preset_params(preset_UID)
        # TODO multi-wav (multi MIDI note) loading
        x_wav, _ = self.get_wav_file(preset_UID, midi_note, midi_velocity)
        # Spectrogram, or Mel-Spectrogram if requested
        spectrogram = self.spectrogram(x_wav)
        if self.spectrogram_normalization == 'min_max':  # result in [-1, 1]
            spectrogram = -1.0 + (spectrogram - self.spec_stats['min'])\
                          / ((self.spec_stats['max'] - self.spec_stats['min']) / 2.0)
        elif self.spectrogram_normalization == 'mean_std':
            spectrogram = (spectrogram - self.spec_stats['mean']) / self.spec_stats['std']
        # Tuple output. Warning: torch.from_numpy does not copy values (torch.tensor(...) ctor does)
        # We add a first dimension to the spectrogram, which is a 1-ch 'greyscale' image
        # TODO multi-channel spectrograms with multiple MIDI notes (and velocities?)
        return torch.unsqueeze(spectrogram, 0), \
            torch.tensor(preset_params[self.learnable_params_idx], dtype=torch.float32), \
            torch.tensor([preset_UID, midi_note, midi_velocity], dtype=torch.int32), \
            self.get_labels_tensor(preset_UID)

    @property
    @abstractmethod
    def total_nb_presets(self):
        pass

    @abstractmethod
    def get_preset_params(self, preset_UID):
        """ Returns the list or numpy array of parameters for the requested preset_UID. Actual implementation
        depends on the synth (some params may be locked or unused). """
        pass

    @property
    @abstractmethod
    def preset_param_names(self):
        """ Returns a List which contains the name of all parameters of presets (free and constrained). """
        pass

    @property
    def learnable_params_count(self):
        """ Returns the length of the second tensor returned by this dataset. """
        return len(self.learnable_params_idx)

    @property
    @abstractmethod
    def total_nb_params(self):
        """ Returns the total count of constrained and free parameters of a preset. """
        pass

    def get_labels_tensor(self, preset_UID):
        """ Returns a tensor of torch.int8 zeros and ones - each value is 1 if the preset is tagged with the
        corresponding label """
        return torch.tensor([1], dtype=torch.int8)  # 'NoLabel' is the only default label

    def get_labels_name(self, preset_UID):
        """ Returns the list of string labels assigned to a preset """
        return ['NoLabel']  # Default: all presets are tagged with this dummy label. Implement in concrete class

    @property
    def available_labels_names(self):
        """ Returns a list of string description of labels. """
        return ['NoLabel']  # this dataset allows no label

    @property
    def labels_count(self):
        return len(self.available_labels_name)

    @abstractmethod
    def get_wav_file(self, preset_UID, midi_note, midi_velocity):
        pass

    def _get_wav_file(self, preset_UID):
        """ Returns the preset_UID audio (numpy array). MIDI note and velocity of the note are the class defaults. """
        # FIXME incompatible with future multi-MIDI notes input
        return self.get_wav_file(preset_UID, self.midi_note, self.midi_velocity)

    def _load_spectrogram_stats(self):
        """ To be called by the child class, after this parent class construction (because stats file path
        depends on child class constructor arguments). """
        try:
            f = open(self._get_spectrogram_stats_file(), 'r')
            self.spec_stats = json.load(f)
        except IOError:
            self.spec_stats = None
            self.spectrogram_normalization = None  # Normalization disabled
            print("[PresetDataset] Cannot open '{}' stats file.".format(self._get_spectrogram_stats_file()))
            print("[PresetDataset] No pre-computed spectrogram stats can be found. No normalization will be performed")

    def get_spectrogram_tensor_size(self):
        """ Returns the size of the first tensor (2D image) returned by this dataset. """
        dummy_spectrogram, _, _, _ = self.__getitem__(0)
        return dummy_spectrogram.size()

    @staticmethod
    def _get_spectrogram_stats_folder():
        """ Returns the path of a './stats' directory inside this script's directory """
        return pathlib.Path(__file__).parent.joinpath('stats')

    def _get_spectrogram_stats_file_stem(self):
        """ Returns the spectrogram stats file base name (without path, suffix and extension) """
        stem = '{}Dataset_spectrogram_nfft{:04d}hop{:04d}mels'.format(self.synth_name, self.n_fft, self.fft_hop)
        stem += ('None' if self.n_mel_bins <= 0 else '{:04d}'.format(self.n_mel_bins))
        return stem

    def _get_spectrogram_stats_file(self):
        return self._get_spectrogram_stats_folder().joinpath(self._get_spectrogram_stats_file_stem() + '.json')

    def _get_spectrogram_full_stats_file(self):
        return self._get_spectrogram_stats_folder().joinpath(self._get_spectrogram_stats_file_stem() + '_full.csv')

    def denormalize_spectrogram(self, spectrogram):
        if self.spectrogram_normalization == 'min_max':  # result in [-1, 1]
            # TODO check this for audio reconstruction
            return (spectrogram + 1.0) * ((self.spec_stats['max'] - self.spec_stats['min']) / 2.0)\
                   + self.spec_stats['min']
        elif self.spectrogram_normalization == 'mean_std':
            return spectrogram * self.spec_stats['std'] + self.spec_stats['mean']

    # TODO un-mel method
    def compute_and_store_spectrograms_stats(self):
        """ Compute min,max,mean,std on all presets previously rendered as wav files.
        Per-preset results are stored into a .csv file
        and dataset-wide averaged results are stored into a .json file

        This functions must be re-run when spectrogram parameters are changed. """
        full_stats = {'UID': [], 'min': [], 'max': [], 'mean': [], 'var': []}
        for i in range(len(self)):
            # We use the exact same spectrogram as the dataloader will
            # TODO multi-midi-notes spectrograms stats
            x_wav, Fs = self.get_wav_file(self.valid_preset_UIDs[i],
                                          self.midi_note, self.midi_velocity)
            assert Fs == 22050
            tensor_spectrogram = self.spectrogram(x_wav)
            full_stats['UID'].append(self.valid_preset_UIDs[i])
            full_stats['min'].append(torch.min(tensor_spectrogram).item())
            full_stats['max'].append(torch.max(tensor_spectrogram).item())
            full_stats['var'].append(torch.var(tensor_spectrogram).item())
            full_stats['mean'].append(torch.mean(tensor_spectrogram, dim=(0, 1)).item())
            if i % 5000 == 0:
                print("Processed stats of {}/{} spectrograms".format(i, len(self)))
        for key in full_stats:
            full_stats[key] = np.asarray(full_stats[key])
        # Average of all columns (std: sqrt(variance avg))
        dataset_stats = {'min': full_stats['min'].min(),
                         'max': full_stats['max'].max(),
                         'mean': full_stats['mean'].mean(),
                         'std': np.sqrt(full_stats['var'].mean()) }
        full_stats['std'] = np.sqrt(full_stats['var'])
        del full_stats['var']
        # Final output
        if not os.path.exists(self._get_spectrogram_stats_folder()):
            os.makedirs(self._get_spectrogram_stats_folder())
        full_stats = pd.DataFrame(full_stats)
        full_stats.to_csv(self._get_spectrogram_full_stats_file())
        with open(self._get_spectrogram_stats_file(), 'w') as f:
            json.dump(dataset_stats, f)
        print("Results written to {} _full.csv and .json files".format(self._get_spectrogram_stats_file_stem()))


