"""
Utilities for plotting various figures (spectrograms, ...)
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa.display


def plot_spectrograms(specs_GT, specs_recons=None, presets_UIDs=None, print_info=False,
                      plot_error=False, error_magnitude=1.0, max_nb_specs=4, spec_ax_w=2.5, spec_ax_h=2.5):
    """
    Creates a figure and axes to plot some ground-truth spectrograms (1st row) and optional reconstructed
    spectrograms (2nd row)

    :returns: fig, axes

    :param specs_GT: Tensor: batch of ground-truth 1-channel spectrograms
    :param specs_recons: Tensor: batch of reconstructed spectrograms
    :param presets_UIDs: 1d-Tensor of preset UIDs corresponding to given spectrograms
    :param error_magnitude: Max error magnitude (used to set the error spectrogram colorbar limits to -mag, +mag)
    :param spec_ax_w: width (in figure units) of a single spectrogram
    """
    nb_specs = np.minimum(max_nb_specs, specs_GT.size(0))
    if specs_recons is None:
        assert plot_error is False  # Cannot plot error without a reconstruction to be compared
        fig, axes = plt.subplots(1, nb_specs, figsize=(nb_specs*spec_ax_w, spec_ax_h))
        axes = [axes]  # Unsqueeze
        nb_rows = 1
    else:
        nb_rows = 2 if not plot_error else 3
        fig, axes = plt.subplots(nb_rows, nb_specs, figsize=(nb_specs*spec_ax_w, spec_ax_h*nb_rows))
    for row in range(nb_rows):
        for i in range(nb_specs):
            if row == 0:
                spectrogram = specs_GT[i, 0, :, :].cpu().numpy()
            elif row == 1:
                spectrogram = specs_recons[i, 0, :, :].cpu().numpy()
            else:
                spectrogram = specs_recons[i, 0, :, :].cpu().numpy() - specs_GT[i, 0, :, :].cpu().numpy()
            UID = presets_UIDs[i].item() if presets_UIDs is not None else None
            if print_info:
                if i == 0:
                    print("Dataset Spectrogram size: {}x{} = {} pixels\nOriginal raw audio: {} samples (22.050kHz, 5.0s))"
                          .format(spectrogram.shape[0], spectrogram.shape[1],
                                  spectrogram.shape[0] * spectrogram.shape[1], 5 * 22050))
                print("Dataset STFT Spectrogram UID={}: min={:.1f} max={:.1f} (normalized dB)"
                      .format(UID, spectrogram.min(), spectrogram.max()))
            if row == 0 and UID is not None:
                axes[row][i].set(title="{}".format(UID))
            librosa.display.specshow(spectrogram, shading='flat', ax=axes[row][i],
                                     cmap=('magma' if row < 2 else 'bwr'),
                                     vmin=(-error_magnitude if row ==2 else None),
                                     vmax=(error_magnitude if row == 2 else None))

    fig.tight_layout()
    return fig, axes

