"""
Audio utils (spectrograms, G&L phase reconstruction, ...)
"""

import warnings

import torch
import torch.fft
import librosa


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

"""Si on veut environ 100dB de dynamic range sur tout le dataset, il semble qu'il faut fixer le min absolu à environ -140dB.

Dexed : le max absolu d'un pixel de spectrogramme est environ -30dB sur le dataset complet, ce qui paraît faible mais est juste. La classe Spectrogram donne bien environ -6dB pour le max du spectre (fréq. positives) d'un sinus pur."""