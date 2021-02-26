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

class DivaDataset(PresetDataset):
    def __init__(self, note_duration = 3.0, n_fft = 512, fft_hop = 512,
                 midi_note=60, midi_velocity=100, n_mel_bins=-1,
                 normalize_audio=False, spectrogram_min_dB=-120.0, spectrogram_normalization ='min_max'
                 ):
        """
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
        self.valid_preset_UIDs = list(range(self._total_nb_presets))
        self.learnable_params_idx = self.valid_preset_UIDs
        for idx in [0]:
            self.learnable_params_idx.remove(idx)
        #del diva_db
        self._params_cardinality = np.asarray([diva.Diva.get_param_cardinality(idx) for idx in range(self.total_nb_params)])
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

    def get_preset_param_cardinality(self, idx, learnable_representation=True):
        return self._params_cardinality[idx]

    def get_full_preset_params(self, preset_UID):
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
        print(file_path)
        try:
            return sf.read(file_path)
        except RuntimeError:
            raise RuntimeError("[data/dataset.py] Can't open file {}. Please pre-render audio files for this "
                               "dataset configuration.".format(file_path))

    def generate_wav_files(self):
        """ Reads all presets (names, param values, and labels) from .pickle and .txt files
         (see dexed.PresetDatabase.write_all_presets_to_files(...)) and renders them
         using attributes and constraints of this class (midi note, normalization, etc...)

         Floating-point .wav files will be stored in dexed presets' folder (see synth/dexed.py)

         Also writes a audio_render_constraints.json file that should be checked when loading data.
         """
        # TODO multiple midi notes generation
        midi_note, midi_velocity = self.midi_note, self.midi_velocity
        for i in range(len(self)):   # TODO full dataset
            preset_UID = self.valid_preset_UIDs[i]
            # Constrained params (1-element batch)
            preset_params = self.get_full_preset_params(preset_UID)
            x_wav, Fs = self._render_audio(preset_params, midi_note, midi_velocity)  # Re-Loads the VST
            sf.write(self.get_wav_file_path(preset_UID, midi_note, midi_velocity), x_wav, Fs, subtype='FLOAT')
            if i % 50 == 0:
                print("Writing .wav files... ({}/{})".format(i, len(self)))
        print("Finished writing {} .wav files".format(len(self)))

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
        preset_UID = self.valid_preset_UIDs[i]

        preset_params = self.get_full_preset_params(preset_UID)
        # TODO multi-wav (multi MIDI note) loading (or rendering... not implemented yet)
        x_wav, _ = self.get_wav_file(preset_UID, midi_note, midi_velocity)
        # Spectrogram, or Mel-Spectrogram if requested (see self.spectrogram ctor arguments)
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
            torch.tensor([preset_UID, midi_note, midi_velocity], dtype=torch.int32)


if __name__ == "__main__":
    # TODO regenerate audio and spectrogram stats files
    # ============== DATA RE-GENERATION - FROM config.py ==================
    regenerate_wav = False  # quite long (15min, full dataset, 1 midi note)
    regenerate_spectrograms_stats = False  # approx 3 min

    # No label restriction, no normalization, etc...
    # But: OPERATORS LIMITATIONS and DEFAULT PARAM CONSTRAINTS (main params (filter, transpose,...) are constant)
    diva_dataset = DivaDataset(note_duration = 3.0, n_fft = 512, fft_hop = 512,
                               midi_note=60, midi_velocity=100, n_mel_bins=-1,
                               normalize_audio=False, spectrogram_min_dB=-120.0,
                               spectrogram_normalization ='min_max'
                               )
    for i in range(2):
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
