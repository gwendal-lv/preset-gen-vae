"""
Implementation of the DivaDataset, based on the PresetBased abstract class.
TODO doc
"""

from abstractbasedataset import PresetDataset
from synth import diva
import numpy as np
import pathlib
import torch
import torch.utils
import soundfile as sf
import json
from multiprocessing import *
import multiprocessing
from datetime import datetime
from data.preset import PresetsParams, PresetIndexesHelper


class DivaDataset(PresetDataset):
    def __init__(self, note_duration = 3.0, n_fft = 512, fft_hop = 512,
                 midi_note=60, midi_velocity=100, n_mel_bins=-1,
                 normalize_audio=False, spectrogram_min_dB=-120.0, spectrogram_normalization ='min_max',
                 constant_filter_and_tune_params=True):
        """
        :param vst_params_learned_as_categorical:
        :param note_duration: Tuple: MIDI Note (on_duration, off_duration) in seconds
        :param n_fft: Width of the FFT window for spectrogram computation
        :param fft_hop: STFT hop length (in samples)
        :param midi_note:  Default dataset MIDI note value (used to pre-render and to load .wav file)
        :param midi_velocity:  Default dataset MIDI velocity value (used to pre-render and to load .wav file)
        :param normalize_audio:  If True, audio from RenderMan will be normalized
        :param spectrogram_min_dB:  Noise-floor threshold value for log-scale spectrograms
        :param spectrogram_normalization: 'min_max' to get output spectrogram values in [-1, 1], or 'mean_std'
            to get zero-mean unit-variance output spectrograms. None to disable normalization.
        """
        super().__init__(note_duration, n_fft, fft_hop, midi_note, midi_velocity, n_mel_bins,
                         normalize_audio, spectrogram_min_dB, spectrogram_normalization)

        self.diva_db = diva.PresetDatabase()
        self._total_nb_presets = self.diva_db.get_nb_presets()
        self._total_nb_params = self.diva_db.get_nb_params()
        self._param_names = self.diva_db.get_param_names()
        self.valid_preset_UIDs = list(range(0, self._total_nb_presets))
        self.learnable_params_idx = list(range(0, self._total_nb_params))
        if constant_filter_and_tune_params is True:
            for vst_idx in [0, 13, 17, 39, 50, 143, 174, 175, 261, 262]:
                self.learnable_params_idx.remove(vst_idx)
        self.diva_synth = diva.Diva()
        self._vst_param_learnable_model = list()
        for vst_idx in range(self.total_nb_params):
            if vst_idx not in self.learnable_params_idx:
                self._vst_param_learnable_model.append(None)
            else:
                param_cardinality = self.diva_synth.get_param_cardinality(vst_idx)
                if param_cardinality > 0:
                    self._vst_param_learnable_model.append('cat')
                else:
                    self._vst_param_learnable_model.append('num')
        # - - - Final initializations - - -
        self._params_cardinality = np.asarray([diva.Diva.get_param_cardinality(idx) for idx in range(self.total_nb_params)])
        self._preset_idx_helper = PresetIndexesHelper(dataset=self, nb_params=self._total_nb_presets)
        self._load_spectrogram_stats()  # Must be called after super() ctor

    @property
    def synth_name(self):
        return "Diva"

    def __str__(self):
        return "Dataset of {}/{} {} presets. {} learnable synth params, {} fixed params. " \
               "size={}, min={:.1f}dB, normalization:{}" \
            .format(len(self), self.total_nb_presets, self.synth_name, len(self.learnable_params_idx),
                    self.total_nb_params - len(self.learnable_params_idx),
                    ("Linear" if self.n_mel_bins <= 0 else "Mel"),
                    self.spectrogram.min_dB, self.spectrogram_normalization)

    def __len__(self):  # Required for any torch.utils.data.Dataset
        return len(self.valid_preset_UIDs)

    @property
    def total_nb_presets(self):
        return self._total_nb_presets

    @property
    def total_nb_params(self):
        return self._total_nb_params

    @property
    def preset_param_names(self):
        return self._param_names

    @property
    def numerical_vst_params(self):
        return diva.Diva.get_numerical_params_indexes()

    @property
    def categorical_vst_params(self):
        return diva.Diva.get_categorical_params_indexes()

    @property
    def vst_param_learnable_model(self):
        return self._vst_param_learnable_model

    def get_preset_param_cardinality(self, idx, learnable_representation=True):
        return self._params_cardinality[idx]

    def get_full_preset(self, preset_UID):
        return self.diva_db.get_preset_values(preset_UID)

    def _render_audio(self, preset_params, midi_note, midi_velocity):
        """ Renders audio on-the-fly and returns the computed audio waveform and sampling rate.

        :param preset_params: Constrained preset parameters (constraints from this class ctor args must have
            been applied before passing preset_params).
        """
        # reload the VST to prevent hanging notes/sounds
        diva_renderer = diva.Diva()
        diva_renderer.assign_preset(preset_params)
        diva_renderer.set_default_general_filter_and_tune_params()
        x_wav = diva_renderer.render_note(midi_note, midi_velocity, normalize=self.normalize_audio)
        return x_wav, diva_renderer.Fs

    def get_wav_file_path(self, preset_UID, midi_note, midi_velocity):
        """ Returns the path of a wav (from diva_presets folder). Operators"""
        presets_folder = diva.PresetDatabase._get_presets_folder()
        filename = "Rendu/preset{:06d}_midi{:03d}vel{:03d}.wav".format(preset_UID, midi_note, midi_velocity)
        return presets_folder.joinpath(filename)

    def get_wav_file(self, preset_UID, midi_note, midi_velocity):
        file_path = self.get_wav_file_path(preset_UID, midi_note, midi_velocity)
        try:
            return sf.read(file_path)
        except RuntimeError:
            raise RuntimeError("[data/dataset.py] Can't open file {}. Please pre-render audio files for this "
                               "dataset configuration.".format(file_path))

    def Configure_parameters(self, constant_filter_and_tune_params=True, osc_param_off=True, dry_param_off=True, scope_param_off=True, fx_param_off=True):
        new_raw_presets = []
        preset_values = []
        for preset in self.diva_db.all_presets_raw:
            self.diva_synth.current_preset = self.diva_db.get_params_in_plugin_format(self.diva_db, preset)
            if constant_filter_and_tune_params is True:
                self.diva_synth.set_default_general_filter_and_tune_params()
            if osc_param_off is True:
                self.diva_synth.set_osc_params_off()
            if dry_param_off is True:
                self.diva_synth.set_dry_params_off()
            if scope_param_off is True:
                self.diva_synth.set_scope_params_off()
            if fx_param_off is True:
                self.diva_synth.set_fx_params_off()
            for el in self.diva_synth.current_preset:
                preset_values.append(el[1])
            new_raw_presets.append(list(preset_values))
            preset_values.clear()

        self.diva_db.all_presets_raw = new_raw_presets

    def generate_wav_files(self):
        """ Reads all presets (names, param values, and labels) from .pickle and .txt files
         (see dexed.PresetDatabase.write_all_presets_to_files(...)) and renders them
         using attributes and constraints of this class (midi note, normalization, etc...)

         Floating-point .wav files will be stored in dexed presets' folder (see synth/dexed.py)

         Also writes a audio_render_constraints.json file that should be checked when loading data.
         """
        # TODO multiple midi notes generation
        midi_note, midi_velocity = self.midi_note, self.midi_velocity
        for i in range(0, len(self)):   # TODO full dataset
            preset_UID = self.valid_preset_UIDs[i]
            # Constrained params (1-element batch)
            preset_params = self.get_full_preset(preset_UID)
            x_wav, Fs = self._render_audio(preset_params, midi_note, midi_velocity)  # Re-Loads the VST
            sf.write(self.get_wav_file_path(preset_UID, midi_note, midi_velocity), x_wav, Fs, subtype='FLOAT')
            if i % 50 == 0:
                print("Writing .wav files... ({}/{})".format(i, len(self)))
        print("Finished writing {} .wav files".format(len(self)))

    def generate_wav_files_multi_process(self, begin, end, process_name):
        midi_note, midi_velocity = self.midi_note, self.midi_velocity
        for i in range(begin, end):
            preset = self.valid_preset_UIDs[i]
            preset_params = self.get_full_preset(self.valid_preset_UIDs[i])
            x_wav, Fs = self._render_audio(preset_params, midi_note, midi_velocity)
            sf.write(self.get_wav_file_path(preset, midi_note, midi_velocity), x_wav, Fs, subtype='FLOAT')
            print("\n")
            print(process_name)
            print(i)
            if (i - begin) % 50 == 0:
                print(process_name + " : Writing .wav files... ({}/{})".format((i - begin), (end - begin)))
        print(process_name + " : Finished writing {} .wav files".format((end - begin)))
        print(datetime.now())

    def get_full_preset_params(self, preset_UID):
        # TODO FAIRE DES ECHANTILLONS
        raw_full_preset = self.diva_db.all_presets_raw
        flatten_list = [j for sub in raw_full_preset for j in sub]
        tensor_2d = torch.unsqueeze(torch.tensor(flatten_list, dtype=torch.float32), 0)
        return PresetsParams(dataset=self, full_presets=tensor_2d, learnable_presets=None)

    def multi_run_wrapper(self, args):
        return self.generate_wav_files_multi_process(*args)

if __name__ == "__main__":
    # ============== DATA RE-GENERATION - FROM config.py ==================
    regenerate_wav = False  # quite long (15min, full dataset, 1 midi note)
    regenerate_wav_multi_process = False
    regenerate_spectrograms_stats = False  # approx 3 min

    # No label restriction, no normalization, etc...
    # But: OPERATORS LIMITATIONS and DEFAULT PARAM CONSTRAINTS (main params (filter, transpose,...) are constant)
    diva_dataset = DivaDataset(note_duration=3.0, n_fft=512, fft_hop=512, midi_note=60, midi_velocity=100,
                               n_mel_bins=-1, normalize_audio=False, spectrogram_min_dB=-120.0,
                               spectrogram_normalization='min_max')
    diva_dataset.Configure_parameters()

    for i in range(1):
        test = diva_dataset[i]  # try get an item - for debug purposes
        print(diva_dataset)  # All files must be pre-rendered before printing
        print(test)

    if regenerate_wav_multi_process:
        num_processor = multiprocessing.cpu_count()
        print(multiprocessing.cpu_count())
        with Pool(num_processor) as p:
            p.map(diva_dataset.multi_run_wrapper, [(0, 2807,"Process 1"), (2808, 5613,"Process 2"), (5614, 8420,"Process 3"), (8421, 11226,"Process 4")])
            print(p)

    if regenerate_wav:
        # WRITE ALL WAV FILES (approx. 10.5Go for 4.0s audio, 1 midi note)
        diva_dataset.generate_wav_files()
    if regenerate_spectrograms_stats:
        # whole-dataset stats (for proper normalization)
        diva_dataset.compute_and_store_spectrograms_stats()
    # ============== DATA RE-GENERATION - FROM config.py ==================
    print("OK")
