"""
Dataset of audio files that were not synthesized from a known synth or preset dataset.

Eventually, this should be merged somehow into abstractbasedataset.py
"""

import torch
import torch.utils
import numpy as np
from typing import Optional, List
from pathlib import Path
import soundfile as sf
import warnings

import utils.audio


class AudioDataset(torch.utils.data.Dataset):
    """
    This class loads all audio files to get min/max stats during construction.
    Thus, it's not adapted to large datasets at the moment.

    Audio files must be named BASENAME_xxx_yyy.wav where xxx and yyy are the MIDI pitch and velocity.
    All available files will be automatically retrieved during construction.
    """

    def __init__(self, midi_notes=((60, 85),), audio_files_folder: Optional[Path] = None,
                 n_samples=88576, sr=22050, fadeout_ms=100.0,
                 n_fft=1024, fft_hop=256, spectrogram_min_dB=-120.0, n_mel_bins=257):
        """

        :param midi_notes:
        :param audio_files_folder:
        :param n_samples: Target number of samples for a given audio file. A small difference between read files and
            this target number can be tolerated. Default value is a bit more than 4.0s 22,05kHz audio (RenderMan
            sends a bit too much samples but Ableton Live for instance provides exactly 88200 samples for 4.0s audio).
        :param fadeout_ms:
        :param n_fft: STFT window length
        :param fft_hop: STFT hop length
        :param spectrogram_min_dB:
        :param n_mel_bins: If -1, the usual STFT will be computed  TODO check this
        """
        self.midi_notes = midi_notes
        # Search for files
        if audio_files_folder is not None:
            self._folder = audio_files_folder
        else:
            self._folder = Path(__file__).resolve().parent.joinpath("external_audio")
        files = [f for f in self._folder.glob('*.wav')]

        # TODO load all appropriate audio files
        #     and check for length consistency
        self._audio_names = list()  # type: List[str]
        self._audio_wav = list()  # type: List[List[np.ndarray]]
        self._files_not_read = list()  # type: List[str]
        self._sr = sr
        self._fadeout_n_samples = int(round(0.001 * fadeout_ms * self._sr))
        for f in files:
            name = f.name.replace('.wav', '')
            # Should we actually read this file or not
            should_read = True
            name_split = name.split('_')  # base audio name might contain some '_'
            if len(name_split) < 3:
                warnings.warn("File name '{}' is not properly formatted (must include MIDI pitch "
                              "and vel e.g. basename_060_127.wav".format(name))
                should_read = False
            try:
                midi_pitch = int(name_split[-2])
            except ValueError:
                warnings.warn("Cannot parse MIDI pitch from file '{}'".format(name))
                should_read = False
            try:
                midi_vel = int(name_split[-1])
            except ValueError:
                warnings.warn("Cannot parse MIDI velocity from file '{}'".format(name))
                should_read = False
            name = '_'.join([name_split[i] for i in range(len(name_split) - 2)])
            if should_read:
                try:  # We do something only if this note is included
                    note_index = self.midi_notes.index((midi_pitch, midi_vel))
                    if name not in self._audio_names:  # maybe add a new name (and associated list to store .wav files)
                        self._audio_names.append(name)
                        self._audio_wav.append([None for _ in range(len(self.midi_notes))])
                    audio_name_index = self._audio_names.index(name)
                    audio_wav, audio_sr = sf.read(f)
                    assert len(audio_wav.shape) == 1  # mono files only
                    # assert sampling freqency
                    if self._sr != audio_sr:
                        raise ValueError("Sampling rate of file '{}' is {}Hz but "
                                         "files must be sampled at {}Hz".format(f, audio_sr, self._sr))
                    # handle small length differences - raise exception is diff is too big
                    if audio_wav.shape[0] < (0.95 * n_samples):
                        raise ValueError("File '{}' is more than 5% too short (number of samples={}, target={}"
                                         .format(f, audio_wav.shape[0], n_samples))
                    elif audio_wav.shape[0] > (1.05 * n_samples):
                        raise ValueError("File '{}' is more than 5% too long (number of samples={}, target={}"
                                         .format(f, audio_wav.shape[0], n_samples))
                    # fadeout and/or zero-padding
                    if self._fadeout_n_samples > 1:
                        fadeout_arr = np.linspace(1.0, 0.0, self._fadeout_n_samples)
                        audio_wav[-self._fadeout_n_samples:] = audio_wav[-self._fadeout_n_samples:] * fadeout_arr
                    if audio_wav.shape[0] < n_samples:
                        audio_wav = np.pad(audio_wav, pad_width=(0, n_samples - audio_wav.shape[0]))
                    self._audio_wav[audio_name_index][note_index] = audio_wav
                except ValueError:  # if this file note is excluded from our dataset
                    self._files_not_read.append(str(f))
                    continue
            else:
                self._files_not_read.append(str(f))

        # Check that all MIDI notes are there (exception raised for missing files)
        for i, name in enumerate(self._audio_names):
            for note_index, audio_wav in enumerate(self._audio_wav[i]):
                if audio_wav is None:
                    raise FileNotFoundError("MIDI note {} is missing for audio files '{}'"
                                            .format(self.midi_notes[note_index], name))

        # compute unnormalized spectrograms
        self._spectrograms = list()  # type: List[torch.Tensor]
        if n_mel_bins < 0:
            raise NotImplementedError("TODO implement STFT ctor arg")
        spectrogram = utils.audio.MelSpectrogram(n_fft, fft_hop, spectrogram_min_dB, n_mel_bins, self._sr)
        for i, _ in enumerate(self._audio_names):
            tensor_specs = [spectrogram(a) for a in self._audio_wav[i]]
            self._spectrograms.append(torch.stack(tensor_specs))

        # normalize spectrograms into [-1, 1] and keep them stored
        s_min = min([s_stack.min() for s_stack in self._spectrograms])
        s_max = max([s_stack.max() for s_stack in self._spectrograms])
        self._spectrograms = [2.0 * (s_stack - s_min) / (s_max - s_min) - 1.0
                              for s_stack in self._spectrograms]

    def __str__(self):
        return "[{}] Loaded {} files ({} different names, {} MIDI note(s) per name) from {}. Spectrograms shape: {}"\
            .format(self.__class__.__name__, len(self._audio_names) * len(self.midi_notes),
                    len(self._audio_names), len(self.midi_notes), self._folder, self._spectrograms[0].shape)

    def __len__(self):
        return len(self._audio_names)

    def __getitem__(self, idx):
        return self.get_spectrogram(idx)

    @property
    def sampling_rate(self):
        return self._sr

    @property
    def nb_midi_notes(self):
        return len(self.midi_notes)

    def get_audio(self, idx):
        """ Returns a list of audio arrays (multiple notes) for the given audio instrument index. """
        return self._audio_wav[idx]

    def get_audio_name(self, idx):
        return self._audio_names[idx]

    def get_index_from_name(self, name):
        return self._audio_names.index(name)

    def get_spectrogram(self, idx):
        return self._spectrograms[idx]



if __name__ == "__main__":
    audio_dataset = AudioDataset(midi_notes=((60, 85), ))
    print(audio_dataset)
