""" TODO proper doc """

import os
import pathlib
from abc import ABC, abstractmethod  # Abstract Base Class
import pandas as pd
import json
from datetime import datetime
import multiprocessing

import torch
import torch.utils
import numpy as np

import utils.audio

from data.preset import PresetsParams, PresetIndexesHelper

# See https://github.com/pytorch/audio/issues/903
#torchaudio.set_audio_backend("sox_io")


class PresetDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, note_duration,
                 n_fft, fft_hop,  # ftt 1024 hop=512: spectrogram is approx. the size of 5.0s@22.05kHz audio
                 midi_notes=((60, 100),),
                 multichannel_stacked_spectrograms=False,
                 n_mel_bins=-1, mel_fmin=30.0, mel_fmax=11e3,
                 normalize_audio=False, spectrogram_min_dB=-120.0, spectrogram_normalization='min_max',
                 learn_mod_wheel_params=False
                 ):
        """
        Abstract Base Class for any synthesizer presets dataset.

        :param note_duration: Tuple: MIDI Note (on_duration, off_duration) in seconds
        :param n_fft: Width of the FFT window for spectrogram computation
        :param fft_hop: STFT hop length (in samples)
        :param midi_notes: Tuple of (midi_pitch, midi_velocity) tuples of notes that should be rendered. Length
            of this tuple is the number of spectrograms that will be fed to the encoder.
        :param multichannel_stacked_spectrograms: If True, this dataset will multi-layer spectrograms
            (1 layer = 1 midi pitch and velocity). If False, the dataset length will be multiplied by the number
            of midi notes.
        :param n_mel_bins: Number of frequency bins for the Mel-spectrogram. If -1, the normal STFT will be used
        :param mel_fmin: TODO implement
        :param mel_fmax: TODO implement
        :param normalize_audio:  If True, audio from RenderMan will be normalized
        :param spectrogram_min_dB:  Noise-floor threshold value for log-scale spectrograms
        :param spectrogram_normalization: 'min_max' to get output spectrogram values in [-1, 1], or 'mean_std'
            to get zero-mean unit-variance output spectrograms. None to disable normalization.
        :param learn_mod_wheel_params: Indicates whether parameters related to the MIDI modulation wheel should
            be learned or not.
        """
        self.note_duration = note_duration
        self.n_fft = n_fft
        self.fft_hop = fft_hop
        self.midi_notes = midi_notes
        if len(self.midi_notes) == 1:  # A 1-note dataset cannot handle multi-note stacked spectrograms
            assert not multichannel_stacked_spectrograms  # Check ctor arguments
        self._multichannel_stacked_spectrograms = multichannel_stacked_spectrograms
        self.n_mel_bins = n_mel_bins
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.normalize_audio = normalize_audio
        self.learn_mod_wheel_params = learn_mod_wheel_params
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
        return "Dataset of {}/{} {} presets. Total items count {}: {} MIDI notes / preset, {} spectrograms.\n" \
               "{} learnable synth params, {} fixed params.\n" \
               "{} Spectrogram items, size={}, min={:.1f}dB, normalization:{}" \
            .format(self.valid_presets_count, self.total_nb_presets, self.synth_name,
                    len(self), self.midi_notes_per_preset,
                    ('stacked' if self.midi_notes_per_preset > 1 and self._multichannel_stacked_spectrograms else 'independent'),
                    len(self.learnable_params_idx), self.total_nb_params - len(self.learnable_params_idx),
                    ("Linear" if self.n_mel_bins <= 0 else "Mel"), self.get_spectrogram_tensor_size(),
                    self.spectrogram.min_dB, self.spectrogram_normalization)

    def __len__(self):  # Required for any torch.utils.data.Dataset
        if self._multichannel_stacked_spectrograms:
            return self.valid_presets_count
        else:
            return self.valid_presets_count * self.midi_notes_per_preset

    def __getitem__(self, i):
        """ Returns a tuple containing a 2D scaled dB spectrograms tensor
        (1st dim: MIDI note, 2nd dim: freq; 2rd dim: time),
        a 1D tensor of parameter values in [0;1],
        and a 1d tensor with remaining int info (preset UID, midi note, vel).

        If this dataset generates audio directly from the synth, only 1 dataloader is allowed.
        A 30000 presets dataset require approx. 7 minutes to be generated on 1 CPU. """
        # TODO on-the-fly audio generation. We should try:
        #  - Use shell command to run a dedicated script. The script writes AUDIO_SAMPLE_TEMP_ID.wav
        #  - wait for the file to be generated on disk (or for the command to notify... something)
        #  - read and delete this .wav file
        # If several notes available but single-spectrogram output: we have to convert i into a UID and a note index
        if self.midi_notes_per_preset > 1 and not self._multichannel_stacked_spectrograms:
            preset_index = i // self.midi_notes_per_preset
            midi_note_indexes = [i % self.midi_notes_per_preset]
        else:
            preset_index = i
            midi_note_indexes = range(self.midi_notes_per_preset)
        # Load params and a list of spectrograms (1-element list is fine). 1 spectrogram per MIDI
        preset_UID = self.valid_preset_UIDs[preset_index]
        preset_params = self.get_full_preset_params(preset_UID)
        spectrograms = list()
        for midi_note_idx in midi_note_indexes:
            midi_pitch, midi_velocity = self.midi_notes[midi_note_idx]
            x_wav, _ = self.get_wav_file(preset_UID, midi_pitch, midi_velocity)
            # Spectrogram, or Mel-Spectrogram if requested (see self.spectrogram ctor arguments)
            spectrogram = self.spectrogram(x_wav)
            if self.spectrogram_normalization == 'min_max':  # result in [-1, 1]
                spectrogram = -1.0 + (spectrogram - self.spec_stats['min'])\
                              / ((self.spec_stats['max'] - self.spec_stats['min']) / 2.0)
            elif self.spectrogram_normalization == 'mean_std':
                spectrogram = (spectrogram - self.spec_stats['mean']) / self.spec_stats['std']
            spectrograms.append(spectrogram)
        # Tuple output. Warning: torch.from_numpy does not copy values (torch.tensor(...) ctor does)
        # FIXME the MIDI pitch and velocity should be a separate tensor, for multi-layer spectrogram
        #   but this will break compatibility with much code and many notebooks
        if len(midi_note_indexes) == 1:
            ref_midi_pitch, ref_midi_velocity = self.midi_notes[midi_note_indexes[0]]
        else:
            ref_midi_pitch, ref_midi_velocity = self.midi_notes[0]
        return torch.stack(spectrograms), \
            torch.squeeze(preset_params.get_learnable(), 0), \
            torch.tensor([preset_UID, ref_midi_pitch, ref_midi_velocity], dtype=torch.int32), \
            self.get_labels_tensor(preset_UID)

    @property
    @abstractmethod
    def total_nb_presets(self):
        """ Total number of presets in the original database, which might be greater than the number of
        available presets in this dataset (some presets can be excluded from learning). """
        pass

    @property
    def valid_presets_count(self):
        """ Total number of presets currently available from this dataset. """
        return len(self.valid_preset_UIDs)

    @property
    def midi_notes_per_preset(self):
        """ Number of available midi notes (different pitch and/or velocity) for a given preset. """
        return len(self.midi_notes)

    @property
    def multichannel_stacked_spectrograms(self):
        """ If True, this dataset's spectrograms are multi-channel, each channel corresponding to a MIDI note.
         If False, this dataset's spectrograms are single-channel, but different dataset items can correspond to
         different MIDI notes. """
        return self._multichannel_stacked_spectrograms

    @abstractmethod
    def get_full_preset_params(self, preset_UID) -> PresetsParams:
        """ Returns a PresetsParams instance (see preset.py) of 1 preset for the requested preset_UID """
        pass

    @property
    def preset_param_names(self):
        """ Returns a List which contains the name of all parameters of presets (free and constrained). """
        return ['unnamed_param_{}'.format(i) for i in range(self.total_nb_params)]

    def get_preset_param_cardinality(self, idx, learnable_representation=True):
        """ Returns the cardinality i.e. the number of possible different values of all parameters.
        A -1 cardinal indicates a continuous parameter.

        :param idx: The full-preset (VSTi representation) index
        :param learnable_representation: Some parameters can have a reduced cardinality for learning
        (and their learnable representation is scaled consequently). """
        return -1  # Default: continuous params only

    def get_preset_param_quantized_steps(self, idx, learnable_representation=True):
        """ Returns a numpy array of possible quantized values of a discrete parameter. Quantized values correspond
        to floating-point VSTi control values. Returns None if idx refers to a continuous parameter. """
        card = self.get_preset_param_cardinality(idx, learnable_representation)
        if card == -1:
            return None
        elif card == 1:  # Constrained one-value parameter
            return np.asarray([0.5])
        elif card >= 2:
            return np.linspace(0.0, 1.0, endpoint=True, num=card)
        else:
            raise ValueError("Invalid parameter cardinality {}".format(card))

    @property
    def learnable_params_count(self):
        """ Number of learnable VSTi controls. """
        return len(self.learnable_params_idx)

    @property
    def learnable_params_tensor_length(self):
        """ Length of a learnable parameters tensor (contains single-element numerical values and one-hot encoded
        categorical params). """
        _, params, _, _ = self.__getitem__(0)
        return params.shape[0]

    @property
    def vst_param_learnable_model(self):
        """ List of models for full-preset (VSTi-compatible) parameters. Possible values are None for non-learnable
        parameters, 'num' for numerical data (continuous or discrete) and 'cat' for categorical data. """
        return ['num' for _ in range(self.total_nb_params)]  # Default: 'num' only

    @property
    def numerical_vst_params(self):
        """ List of indexes of numerical parameters (whatever their discrete number of values) in the VSTi.
        E.g. a 8-step volume param is numerical, while a LFO shape param is not (it is categorical). The
        learnable model can be different from the VSTi model. """
        return [i for i in range(self.total_nb_params)]  # Default: numerical only

    @property
    def categorical_vst_params(self):
        """ List of indexes of categorical parameters in the VSTi. The learnable model can be different
        from the VSTi model."""
        return []  # Default: no categorical params

    @property
    def params_default_values(self):
        """ Dict of default values of VSTi parameters. Not all indexes are keys of this dict (many params do not
        have a default value). """
        return {}

    @property
    @abstractmethod
    def total_nb_params(self):
        """ Total count of constrained and free VST parameters of a preset. """
        pass

    @property
    def preset_indexes_helper(self):
        """ Returns the data.preset.PresetIndexesHelper instance which helps convert full/learnable presets
        from this dataset. """
        return PresetIndexesHelper(nb_params=self.total_nb_params)  # Default: identity

    def get_labels_tensor(self, preset_UID):
        """ Returns a tensor of torch.int8 zeros and ones - each value is 1 if the preset is tagged with the
        corresponding label. """
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
        t_start = datetime.now()
        # MKL and/or PyTorch do not use hyper-threading, and it gives better results... don't use multi-proc here
        workers_args = self._get_multi_note_workers_args(num_workers=1)
        full_stats = self._compute_spectrogram_stats_batch(workers_args[0])
        # Average of all columns (std: sqrt(variance avg))
        dataset_stats = {'min': full_stats['min'].min(),
                         'max': full_stats['max'].max(),
                         'mean': full_stats['mean'].mean(),
                         'std': np.sqrt(full_stats['var'].mean())}
        full_stats['std'] = np.sqrt(full_stats['var'])
        del full_stats['var']
        # Final output
        if not os.path.exists(self._get_spectrogram_stats_folder()):
            os.makedirs(self._get_spectrogram_stats_folder())
        full_stats = pd.DataFrame(full_stats)
        full_stats.to_csv(self._get_spectrogram_full_stats_file())
        with open(self._get_spectrogram_stats_file(), 'w') as f:
            json.dump(dataset_stats, f)
        delta_t = (datetime.now() - t_start).total_seconds()
        print("Results from {} spectrograms written to {} _full.csv and .json files ({:.1f} minutes total)"
              .format(len(full_stats), self._get_spectrogram_stats_file_stem(), delta_t/60.0))

    def _compute_spectrogram_stats_batch(self, worker_args):
        """ Generates and returns a dict of spectrograms stats
        using the given list of (preset_UID, midi_pitch, midi_vel) tuples. """
        full_stats = {'UID': np.zeros((len(worker_args),), dtype=np.int),
                      'min': np.zeros((len(worker_args),)), 'max': np.zeros((len(worker_args),)),
                      'mean': np.zeros((len(worker_args),)), 'var': np.zeros((len(worker_args),))}
        for i, (preset_UID, midi_pitch, midi_velocity) in enumerate(worker_args):
            x_wav, Fs = self.get_wav_file(preset_UID, midi_pitch, midi_velocity)
            assert Fs == 22050
            # We use the exact same spectrogram as the dataloader will
            tensor_spectrogram = self.spectrogram(x_wav)
            full_stats['UID'][i] = preset_UID
            full_stats['min'][i] = torch.min(tensor_spectrogram).item()
            full_stats['max'][i] = torch.max(tensor_spectrogram).item()
            full_stats['var'][i] = torch.var(tensor_spectrogram).item()
            full_stats['mean'][i] = torch.mean(tensor_spectrogram, dim=(0, 1)).item()
        return full_stats

    def _get_multi_note_workers_args(self, num_workers):
        """
        Divide all notes to be rendered for all presets into lists (1 list of tuples per worker)
        each tuple contains (preset_UID, midi_pitch, midi_vel). We split the all presets UID across
        workers, and all midi notes for that UID are rendered (or analyzed, ...) by the assigned worker

        :returns: a list of lists of (preset_UID, midi_pitch, midi_vel) tuples to provide to a pool of workers
        """
        # This split returns empty arrays if more workers than UIDs - else, evenly split sub-arrays
        split_preset_UIDs = np.array_split(self.valid_preset_UIDs, num_workers)
        workers_args = list()
        for worker_idx, preset_UIDs in enumerate(split_preset_UIDs):
            workers_args.append(list())  # New worker argument
            for preset_UID in preset_UIDs:
                for midi_pitch, midi_vel in self.midi_notes:
                    workers_args[worker_idx].append((preset_UID, midi_pitch, midi_vel))
        return workers_args
