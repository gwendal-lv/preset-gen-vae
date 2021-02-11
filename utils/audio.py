"""
Audio utils (spectrograms, G&L phase reconstruction, ...)
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.fft
import librosa
import librosa.display


class Spectrogram:
    """ Class for dB spectrogram computation from a raw audio waveform.
    The min spectrogram value must be provided.
    The default windowing function is Hann. """
    def __init__(self, n_fft, fft_hop, min_dB, dynamic_range_dB=None):
        self.n_fft = n_fft
        self.fft_hop = fft_hop
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
        return self.linear_to_log_scale(spectrogram)

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
        super().__init__(n_fft, fft_hop, min_dB)
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



