"""
Audio utils (spectrograms, G&L phase reconstruction, ...)
"""

import torch
import torch.fft

import warnings


class Spectrogram:
    """ Class for spectrogram computation from a raw audio waveform. The min spectrogram
    and dynamic range must be provided.
    The default windowing function is Hann. """
    def __init__(self, n_fft, fft_hop, min_dB, dynamic_range_dB):
        self.n_fft = n_fft
        self.fft_hop = fft_hop
        self.min_dB = min_dB
        self.dynamic_range_dB = dynamic_range_dB
        self.window = torch.hann_window(self.n_fft, periodic=False)
        self.spectrogram_norm_factor = torch.fft.rfft(self.window).abs().max().item()

    def __call__(self, x_wav):
        warnings.filterwarnings("ignore", category=UserWarning)
        spectrogram = torch.stft(torch.tensor(x_wav, dtype=torch.float32), n_fft=self.n_fft, hop_length=self.fft_hop,
                                 window=self.window, center=True,
                                 pad_mode='constant', onesided=True, return_complex=True).abs()
        warnings.filterwarnings("default", category=UserWarning)
        # normalization of spectrogram module vs. Hann window weight ; min ; dynamic dB range
        spectrogram = spectrogram / self.spectrogram_norm_factor
        spectrogram = torch.maximum(spectrogram,
                                    torch.ones(spectrogram.size()) * 10 ** (self.min_dB / 20.0))
        spectrogram = 20.0 * torch.log10(spectrogram)
        return torch.maximum(spectrogram, torch.ones(spectrogram.size())
                             * (torch.max(spectrogram) - self.dynamic_range_dB))

