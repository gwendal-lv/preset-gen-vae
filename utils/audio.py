"""
Audio utils (spectrograms, G&L phase reconstruction, ...)
"""

import os
import warnings
from typing import Iterable, Sequence, Optional
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.fft
import librosa
import librosa.display
import soundfile as sf


class Spectrogram:
    """ Class for dB spectrogram computation from a raw audio waveform.
    The min spectrogram value must be provided.
    The default windowing function is Hann. """
    def __init__(self, n_fft, fft_hop, min_dB, dynamic_range_dB=None, log_scale=True):
        self.n_fft = n_fft
        self.fft_hop = fft_hop
        self.log_scale = log_scale
        self.min_dB = min_dB
        self.dynamic_range_dB = dynamic_range_dB
        self.window = torch.hann_window(self.n_fft, periodic=False)
        self.spectrogram_norm_factor = torch.fft.rfft(self.window).abs().max().item()

    def get_stft(self, x_wav):
        """ Returns the complex, non-normalized STFT computed from given audio. """
        warnings.filterwarnings("ignore", category=UserWarning)  # Deprecation warning from PyTorch compiled-code
        spectrogram = torch.stft(torch.tensor(x_wav, dtype=torch.float32), n_fft=self.n_fft, hop_length=self.fft_hop,
                                 window=self.window, center=True,
                                 pad_mode='constant', onesided=True, return_complex=True)
        warnings.filterwarnings("default", category=UserWarning)
        return spectrogram

    def __call__(self, x_wav):
        """ Returns the log-scale spectrogram of x_wav audio, with defined minimum 'floor' dB value. """
        spectrogram = self.get_stft(x_wav).abs()
        # normalization of spectrogram module vs. Hann window weight
        spectrogram = spectrogram / self.spectrogram_norm_factor
        if self.log_scale:
            return self.linear_to_log_scale(spectrogram)
        else:
            return spectrogram

    def linear_to_log_scale(self, spectrogram):
        spectrogram = torch.maximum(spectrogram, torch.ones(spectrogram.size()) * 10 ** (self.min_dB / 20.0))
        return 20.0 * torch.log10(spectrogram)

    def log_to_linear_scale(self, spectrogram):
        """ Reverses the log-scale applied to a Tensor spectrogram built by this class

        :returns: the corresponding usual STFT-amplitude spectrogram """
        stft = torch.pow(10.0, spectrogram/20.0)
        return stft * self.spectrogram_norm_factor

    def linear_to_log_scale_with_dynamic_range(self, spectrogram):
        # TODO remove? Dynamic range might be a bad idea for VAEs... (need to reconstruct the 'floor' value)
        assert self.dynamic_range_dB is not None  # Dynamic range not provided? It might be counterproductive anyway
        spectrogram = torch.maximum(spectrogram, torch.ones(spectrogram.size()) * 10 ** (self.min_dB / 20.0))
        spectrogram = 20.0 * torch.log10(spectrogram)
        return torch.maximum(spectrogram,
                             torch.ones(spectrogram.size()) * (torch.max(spectrogram) - self.dynamic_range_dB))



class MelSpectrogram(Spectrogram):
    def __init__(self, n_fft, fft_hop, min_dB, n_mel_bins, Fs):
        super().__init__(n_fft, fft_hop, min_dB, log_scale=True)
        # TODO add fmin, fmax arguments
        self.Fs = Fs
        self.n_mel_bins = n_mel_bins

    def __call__(self, x_wav):
        """ Returns a log-scale spectrogram with limited dynamic range """
        spectrogram = self.get_stft(x_wav).abs()
        spectrogram = spectrogram / self.spectrogram_norm_factor
        # Torch-Numpy arrays share the same memory location (very fast convert)
        spectrogram = librosa.feature.melspectrogram(S=spectrogram, n_mels=self.n_mel_bins,
                                                     norm=None)  # for linear/mel specs magnitude compatibility
        return self.linear_to_log_scale(torch.from_numpy(spectrogram))

    def mel_dB_to_STFT(self, mel_spectrogram):
        """ Inverses the Mel-filters and and log-amplitude transformations applied to a spectrogram. """
        spectrogram = self.log_to_linear_scale(mel_spectrogram)
        return librosa.feature.inverse.mel_to_stft(spectrogram.numpy(), n_fft=self.n_fft, power=1.0, norm=None)


class SimilarityEvaluator:
    """ Class for evaluating audio similarity between audio samples through various criteria. """
    def __init__(self, x_wav: Sequence[Iterable], n_fft=1024, fft_hop=256, sr=22050, n_mfcc=13):
        """

        :param x_wav: List or Tuple which contains the 2 audio signals (arrays) to be compared.
        :param n_fft:
        :param fft_hop:
        :param sr:
        :param n_mfcc:
        """
        assert len(x_wav) == 2  # This class requires exactly 2 input audio signals
        self.x_wav = x_wav
        self.n_fft = n_fft
        self.fft_hop = fft_hop
        self.sr = sr
        self.n_mfcc = n_mfcc
        # Pre-compute STFT (used in mae log and spectral convergence)
        self.stft = [np.abs(librosa.stft(x, self.n_fft, self.fft_hop)) for x in self.x_wav]

    def get_mae_log_stft(self, return_spectrograms=True):
        """ Returns the Mean Absolute Error on log(|STFT|) spectrograms of input sounds, and the two spectrograms
        themselves (e.g. for plotting them later). """
        eps = 1e-4  # -80dB  (un-normalized stfts)
        log_stft = [np.maximum(s, eps) for s in self.stft]
        log_stft = [np.log10(s) for s in log_stft]
        mae = np.abs(log_stft[1] - log_stft[0]).mean()
        return (mae, log_stft) if return_spectrograms else mae

    def display_stft(self, s, log_scale=True):
        """ Displays given spectrograms s (List of two |STFT|) """
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        im = librosa.display.specshow(s[0], shading='flat', ax=axes[0], cmap='magma')
        im = librosa.display.specshow(s[1], shading='flat', ax=axes[1], cmap='magma')
        if log_scale:
            axes[0].set(title='Reference $\log_{10} |STFT|$')
        else:
            axes[0].set(title='Reference $|STFT|$')
        axes[1].set(title='Inferred synth parameters')
        fig.tight_layout()
        return fig, axes

    def get_spectral_convergence(self, return_spectrograms=True):
        """ Returns the Spectral Convergence of input sounds, and the two linear-scale spectrograms
            used to compute SC (e.g. for plotting them later). SC: see https://arxiv.org/abs/1808.06719 """
        # Frobenius norm is actually the default numpy matrix norm
        # TODO check for 0.0 frob norm of stft[0]
        sc = np.linalg.norm(self.stft[0] - self.stft[1], ord='fro') / np.linalg.norm(self.stft[0], ord='fro')
        return (sc, self.stft) if return_spectrograms else sc

    def get_mae_mfcc(self, return_mfccs=True, n_mfcc: Optional[int] = None):
        """ Returns the Mean Absolute Error on MFCCs, and the MFCCs themselves.
        Uses librosa default MFCCs configuration: TODO precise
        """
        mfcc = [librosa.feature.mfcc(x, sr=self.sr, n_mfcc=(self.n_mfcc if n_mfcc is None else n_mfcc))
                for x in self.x_wav]
        mae = np.abs(mfcc[0] - mfcc[1]).mean()
        return (mae, mfcc) if return_mfccs else mae

    def display_mfcc(self, mfcc):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        im = librosa.display.specshow(mfcc[0], shading='flat', ax=axes[0], cmap='viridis')
        im = librosa.display.specshow(mfcc[1], shading='flat', ax=axes[1], cmap='viridis')
        axes[0].set(title='{}-bands MFCCs'.format(self.n_mfcc))
        axes[1].set(title='Inferred synth parameters')
        fig.tight_layout()
        return fig, axes




class SimpleSampleLabeler:
    def __init__(self, x_wav, Fs, hpss_margin=3.0, perc_duration_ms=250.0):
        """ Class to attribute labels or a class to sounds, mostly based on librosa hpss and empirical thresholds.

        :param x_wav:
        :param Fs:
        :param hpss_margin: see margin arg of librosa.decompose.hpss
        :param perc_duration_ms: The duration of a percussion sound - most of the percussive energy should be found
            before that time (in the percussive separated spectrogram).
        """
        assert Fs == 22050  # Librosa defaults must be used at the moment
        self.x_wav = x_wav
        self.Fs = Fs
        self.hpss_margin = hpss_margin
        self.perc_duration_ms = perc_duration_ms
        # Pre-computation of spectrograms and energies
        self.specs = self._get_hpr_specs()
        self.energy, self.energy_ratio = self._get_energy_ratios()
        # Energies on attack (to identify perc sounds)
        # Perc content supposed to be found in the first 10s of ms. Hop: default librosa 256
        limit_index = int(np.ceil(self.perc_duration_ms * self.Fs / 256.0 / 1000.0))
        self.attack_specs = dict()
        self.attack_energies = dict()
        for k in self.specs:
            self.attack_specs[k] = self.specs[k][:, 0:limit_index]  # indexes: f, t
            self.attack_energies[k] = np.abs(self.attack_specs[k]).sum()
        # Labels pre-computation... so it's done
        self.is_harmonic = self._is_harmonic()
        self.is_percussive = self._is_percussive()

    def has_label(self, label):
        if label == 'harmonic':
            return self.is_harmonic
        elif label == 'percussive':
            return self.is_percussive
        elif label == 'sfx':
            return not self.is_harmonic and not self.is_percussive
        else:
            raise ValueError("Label '{}' is not valid.".format(label))

    def _get_hpr_specs(self):
        D = librosa.stft(self.x_wav)  # TODO custom fft params
        H, P = librosa.decompose.hpss(D, margin=self.hpss_margin)
        R = D - (H + P)
        return {'D': D, 'H': H, 'P': P, 'R': R}

    def _get_energy_ratios(self):
        energy = dict()
        for k in self.specs:
            energy[k] = np.abs(self.specs[k]).sum()
        return energy, {'D': 1.0, 'H': energy['H'] / energy['D'], 'P': energy['P'] / energy['D'],
                        'R': energy['R'] / energy['D']}

    def plot_hpr_specs(self, figsize=(8, 6)):
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        axes = [axes]  # Unqueeze - to prepare for multi-cols display
        for col in range(1):
            im = librosa.display.specshow(librosa.amplitude_to_db(np.abs(self.specs['D']), ref=np.max), y_axis='log',
                                          ax=axes[col][0])
            fig.colorbar(im, format='%+2.0f dB', ax=axes[col][0])
            axes[col][0].set(title='Full power spectrogram')
            im = librosa.display.specshow(librosa.amplitude_to_db(np.abs(self.specs['H']), ref=np.max), y_axis='log',
                                          ax=axes[col][1])
            fig.colorbar(im, format='%+2.0f dB', ax=axes[col][1])
            axes[col][1].set(title='Harmonic power spectrogram ({:.1f}% of total spectral power)'.format(
                100.0 * self.energy_ratio['H']))
            im = librosa.display.specshow(librosa.amplitude_to_db(np.abs(self.specs['P']), ref=np.max), y_axis='log',
                                          ax=axes[col][2])
            fig.colorbar(im, format='%+2.0f dB', ax=axes[col][2])
            axes[col][2].set(title='Percussive power spectrogram ({:.1f}% of total spectral power)'.format(
                100.0 * self.energy_ratio['P']))
            im = librosa.display.specshow(librosa.amplitude_to_db(np.abs(self.specs['R']), ref=np.max), y_axis='log',
                                          ax=axes[col][3])
            fig.colorbar(im, format='%+2.0f dB', ax=axes[col][3])
            axes[col][3].set(title='Residuals power spectrogram ({:.1f}% of total spectral power)'.format(
                100.0 * self.energy_ratio['R']))
        fig.tight_layout()
        return fig, axes

    def get_harmonic_sound(self):
        return librosa.istft(self.specs['H'])

    def get_percussive_sound(self):
        return librosa.istft(self.specs['P'])

    def get_residual_sound(self):
        return librosa.istft(self.specs['R'])

    def _is_harmonic(self):
        if self.energy_ratio['H'] > 0.40:
            return True
        elif self.energy_ratio['H'] > 0.35:  # Harmonic with percussive attack
            return (self.attack_energies['P'] / self.energy['P']) > 0.9
        return False

    def _is_percussive(self):
        # Mostly percussive sound
        if self.energy_ratio['P'] > 0.40:
            return (self.attack_energies['P'] / self.energy['P']) > 0.9
        # Percussive with harmonic attack
        elif self.energy_ratio['P'] > 0.35 and self.energy_ratio['H'] > 0.15:
            return (self.attack_energies['P'] / self.energy['P']) > 0.9\
                   and (self.attack_energies['H'] / self.energy['H']) > 0.8
        return False

    def print_labels(self):
        print("is_harmonic={}   is_percussive={}".format(self.is_harmonic, self.is_percussive))



def write_wav_and_mp3(base_path: pathlib.Path, base_name: str, samples, sr):
    """ Writes a .wav file and converts it to .mp3 using command-line ffmpeg (which must be available). """
    wav_path_str = "{}".format(base_path.joinpath(base_name + '.wav'))
    sf.write(wav_path_str, samples, sr)
    mp3_path_str = "{}".format(base_path.joinpath(base_name + '.mp3'))
    # mp3 320k will be limited to 160k for mono audio - still too much loss for HF content
    os.system("ffmpeg -i {} -b:a 320k -y {}".format(wav_path_str, mp3_path_str))



def get_spectrogram_from_audio(audio_samples_path: pathlib.Path, audio_name: str, midi_notes: Sequence):
    """
    Builds a dataset-like tensor of spectrogram(s) by loading external audio sample(s).
    Files to be loaded must be: BASENAME_xxx_yyy.wav where xxx and yyy are the MIDI pitch and velocity.

    :param audio_samples_path: Path to the folder that contains all audio files to be loaded
    :param audio_name: base name of file(s) to be loaded
    :param midi_notes: List of tuples of (midi_pitch, midi_velocity) int values. Length of that list will be the
        number of output tensor spectrogram channels
    :return: 3d tensor of spectrogram(s)
    """
    pass

