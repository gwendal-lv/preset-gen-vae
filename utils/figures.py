"""
Utilities for plotting various figures (spectrograms, ...)
"""

from typing import Optional
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import librosa.display

import logs.metrics

from data.abstractbasedataset import PresetDataset
from data.preset import PresetIndexesHelper


# Display parameters for scatter/box/error plots
__param_width = 0.12
__x_tick_font_size = 8


def plot_spectrograms(specs_GT, specs_recons=None, presets_UIDs=None, print_info=False,
                      plot_error=False, error_magnitude=1.0, max_nb_specs=4, spec_ax_w=2.5, spec_ax_h=2.5,
                      add_colorbar=False):
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
    if add_colorbar:
        spec_ax_w *= 1.3
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
                spectrogram = specs_GT[i, 0, :, :].clone().detach().cpu().numpy()
            elif row == 1:
                spectrogram = specs_recons[i, 0, :, :].clone().detach().cpu().numpy()
            else:
                spectrogram = specs_recons[i, 0, :, :].clone().detach().cpu().numpy()\
                              - specs_GT[i, 0, :, :].clone().detach().cpu().numpy()
            UID = presets_UIDs[i].item() if presets_UIDs is not None else None
            if print_info:
                if i == 0:
                    print("Dataset Spectrogram size: {}x{} = {} pixels\n"
                          "Original raw audio: {} samples (22.050kHz, 4.0s))"
                          .format(spectrogram.shape[0], spectrogram.shape[1],
                                  spectrogram.shape[0] * spectrogram.shape[1], 4 * 22050))  # TODO don't hardcode
                print("Dataset STFT Spectrogram UID={}: min={:.1f} max={:.1f} (normalized dB)"
                      .format(UID, spectrogram.min(), spectrogram.max()))
            if row == 0 and UID is not None:
                axes[row][i].set(title="{}".format(UID))
            im = librosa.display.specshow(spectrogram, shading='flat', ax=axes[row][i],
                                          cmap=('magma' if row < 2 else 'bwr'),
                                          vmin=(-error_magnitude if row == 2 else None),
                                          vmax=(error_magnitude if row == 2 else None))
            if add_colorbar:
                clb = fig.colorbar(im, ax=axes[row][i], orientation='vertical')

    fig.tight_layout()
    return fig, axes


def plot_latent_distributions_stats(latent_metric: logs.metrics.LatentMetric,
                                    plot_mu=True, plot_sigma=False, figsize=None):
    """ Uses boxplots to represent the distribution of the mu and/or sigma parameters of
    latent gaussian distributions. """
    if plot_sigma or not plot_mu:
        raise NotImplementedError("todo...")
    z_mu = latent_metric.get_z('mu')
    if figsize is None:
        figsize = (__param_width * z_mu.shape[1], 3)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.boxplot(data=z_mu, ax=ax, fliersize=0.3, linewidth=0.5)
    ax.set(xlabel='z', ylabel='$q_{\phi}(z|x) : \mu$')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
        tick.set_fontsize(8)
    fig.tight_layout()
    return fig, ax


def plot_spearman_correlation(latent_metric: logs.metrics.LatentMetric):
    """ Plots the spearman correlation matrix (full, and with zeroed diagonal)
    and returns fig, axes """
    # http://jkimmel.net/disentangling_a_latent_space/ : Uncorrelated (independent) latent variables are necessary
    # but not sufficient to ensure disentanglement...
    corr = latent_metric.get_spearman_corr()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im = axes[0].matshow(corr, cmap='viridis', vmin=-1.0, vmax=1.0)
    clb = fig.colorbar(im, ax=axes[0], orientation='vertical')
    axes[0].set_xlabel('Spearman corr')
    # 0.0 on diagonal - to get a better view on variations (2nd plot)
    corr = latent_metric.get_spearman_corr_zerodiag()
    max_v = np.abs(corr).max()
    im = axes[1].matshow(corr, cmap='viridis', vmin=-max_v, vmax=max_v)
    clb = fig.colorbar(im, ax=axes[1], orientation='vertical')
    axes[1].set_xlabel('zeroed diagonal')
    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
    fig.tight_layout()
    return fig, axes


def _configure_params_plot_x_tick_labels(ax):
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
        tick.set_fontsize(__x_tick_font_size)


def plot_synth_preset_param(ref_preset, inferred_preset=None,
                            preset_UID=None, dataset: Optional[PresetDataset] = None):  # TODO figsize arg
    """ Plots reference parameters values of 1 preset (full VSTi-compatible representation),
    and their corresponding reconstructed values if given.

    :param ref_preset: A ground-truth preset (must be full)
    :param inferred_preset: Reconstructed preset (optional)
    :param dataset: (optional) PresetDataset class, to improve the display (param names, cardinality, ...)
    """
    if inferred_preset is not None:
        assert len(ref_preset) == len(inferred_preset)
    fig, ax = plt.subplots(1, 1, figsize=(__param_width * len(ref_preset), 4))  # TODO dynamic fig size
    # Params cardinality: deduced from the synth arg (str)
    if dataset is not None:
        if dataset.synth_name.lower() == 'dexed':
            # Gets cardinality of *all* params (including non-learnable)
            # Directly ask for quantized values
            params_quant_values = [dataset.get_preset_param_quantized_steps(i, learnable_representation=False)
                                   for i in range(len(ref_preset))]
            for i, y_values in enumerate(params_quant_values):
                if y_values is not None:  # Discrete param
                    marker = '_' if y_values.shape[0] > 1 else 'x'
                    sns.scatterplot(x=[i for _ in range(y_values.shape[0])], y=y_values, marker=marker,
                                    color='grey', ax=ax)
        else:
            raise NotImplementedError("Synth '{}' parameters cannot be displayed".format(dataset.synth_name))
    # For easier seaborn-based plot: we use a pandas dataframe
    df = pd.DataFrame({'param_idx': range(len(ref_preset)), 'ref_preset': ref_preset})
    learnable_param_indexes = dataset.learnable_params_idx if dataset is not None else None
    if learnable_param_indexes is not None:
        df['is_learnable'] = [(idx in learnable_param_indexes) for idx in range(len(ref_preset))]
    else:
        df['is_learnable'] = [True for idx in range(len(ref_preset))]
    # Scatter plot for "faders" values
    sns.scatterplot(data=df, x='param_idx', y='ref_preset', ax=ax,
                    hue="is_learnable",
                    palette=("blend:#BBB,#06D" if learnable_param_indexes is not None else "deep"))
    if inferred_preset is not None:
        df['inferred_preset'] = inferred_preset
        sns.scatterplot(data=df, x='param_idx', y='inferred_preset', ax=ax,
                        hue="is_learnable",
                        palette=("blend:#BBB,#D60" if learnable_param_indexes is not None else "husl"))
    ax.set_xticks(range(len(ref_preset)))
    param_names = dataset.preset_param_names if dataset is not None else None
    ax.set_xticklabels(['{}.{}'.format(idx, ('' if param_names is None else param_names[idx]))
                             for idx in range(len(ref_preset))])
    ax.set(xlabel='', ylabel='Param. value', xlim=[0-0.5, len(ref_preset)-0.5])
    ax.get_legend().remove()
    if preset_UID is not None:
        ax.set_title("Preset UID={} (VSTi numerical parameters)".format(preset_UID))
    # vertical "faders" separator lines
    plt.vlines(x=np.arange(len(ref_preset) + 1) - 0.5, ymin=0.0, ymax=1.0, colors='k', linewidth=1.0)
    _configure_params_plot_x_tick_labels(ax)
    fig.tight_layout()
    return fig, ax


def _get_learnable_preset_xticklabels(idx_helper: PresetIndexesHelper):
    vst_param_names = idx_helper.vst_param_names
    x_tick_labels = list()
    # Param names - vst-param by vst-param. We do not actually care about the learnable index
    for vst_idx, learnable_indexes in enumerate(idx_helper.full_to_learnable):
        if learnable_indexes is not None:  # learnable only
            if isinstance(learnable_indexes, Iterable):  # cat learnable representation
                for i, _ in enumerate(learnable_indexes):
                    if i == 0:
                        x_tick_labels.append('{}.{}.{}'.format(vst_idx, i, vst_param_names[vst_idx]))
                    else:
                        x_tick_labels.append('{}.{}'.format(vst_idx, i))
            else:  # numerical learnable representation
                x_tick_labels.append('{}.{}'.format(vst_idx, vst_param_names[vst_idx]))
    return x_tick_labels


def plot_synth_learnable_preset(learnable_preset, idx_helper: PresetIndexesHelper, preset_UID=None, figsize=None):
    """ Plots a single learnable preset (provided as 1D Tensor) """
    n_params = learnable_preset.size(0)
    assert n_params == idx_helper.learnable_preset_size
    fig, ax = plt.subplots(1, 1, figsize=(__param_width * n_params, 4))  # TODO dynamic fig size
    learnable_param_indexes = range(idx_helper.learnable_preset_size)
    # quantized values - plot now
    params_quant_values = [idx_helper.get_learnable_param_quantized_steps(idx)
                           for idx in learnable_param_indexes]
    for i, y_values in enumerate(params_quant_values):
        if y_values is not None:  # Discrete param only
            marker = '_' if y_values.shape[0] > 1 else 'x'
            sns.scatterplot(x=[i for _ in range(y_values.shape[0])], y=y_values, marker=marker,
                            color='grey', ax=ax)
    # For easier seaborn-based plot: we use a pandas dataframe
    df = pd.DataFrame({'param_idx': range(n_params), 'ref_preset': learnable_preset})
    df['is_learnable'] = [True for _ in range(n_params)]
    # Scatter plot for "faders" values
    sns.scatterplot(data=df, x='param_idx', y='ref_preset', ax=ax)
    ax.set_xticks(range(n_params))
    ax.set_xticklabels(_get_learnable_preset_xticklabels(idx_helper))
    ax.set(xlabel='', ylabel='Param. value', xlim=[0-0.5, n_params-0.5])
    if preset_UID is not None:
        ax.set_title("Preset UID={} (learnable parameters)".format(preset_UID))
    # vertical "faders" separator lines
    plt.vlines(x=np.arange(n_params + 1) - 0.5, ymin=0.0, ymax=1.0, colors='k', linewidth=1.0)
    _configure_params_plot_x_tick_labels(ax)
    fig.tight_layout()
    return fig, ax


def plot_synth_preset_error(param_batch_errors, idx_helper: PresetIndexesHelper,
                            mae_y_limit=0.59, boxplots_y_limits=(-1.1, 1.1),
                            figsize=None):
    """ Uses boxplots to show the error between inferred (out) and GT (in) preset parameters.

    :param mae_y_limit: Constant y-axis upper display limit (to help visualize improvements during training).
        Won't be used is a computed MAE is actually greater than this value.
    :param boxplots_y_limits: Constant y-axis box plots display limits.
    :param param_batch_errors: 2D Tensor of learnable synth parameters error (numerical and categorical)
    :param idx_helper: to improve the display (param names, cardinality, ...) """
    # init
    n_params = param_batch_errors.size(1)
    assert n_params == idx_helper.learnable_preset_size
    batch_errors_np = param_batch_errors.numpy()
    if figsize is None:
        figsize = (__param_width * n_params, 5)
    # Search for synth groups of parameters
    param_groups_separations = []
    if idx_helper.synth_name.lower() == "dexed":
        groups_start_vst_indexes = [23 + 22*i for i in range(6)]  # 23. OP1 EG RATE 1 (1st operator MIDI param)
        cur_group = 0
        for learn_idx in range(n_params):
            # We add a new group when a threshold is reached
            if idx_helper.learnable_to_full[learn_idx] >= groups_start_vst_indexes[cur_group]:
                param_groups_separations.append(learn_idx - 0.5)
                cur_group += 1
    else:
        print("[utils/figures.py] Unknown synth '{}' from given PresetIndexesHelper. "
              "No groups separations displayed on error plot.".format(idx_helper.synth_name))
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    mae = np.abs(batch_errors_np).mean(axis=0)
    for learn_idx in range(batch_errors_np.shape[1]):
        learnable_model = idx_helper.vst_param_learnable_model[idx_helper.learnable_to_full[learn_idx]]
        if learnable_model == 'num':
            color = '#1F77B4'  # mpl C0 - blue
        elif learnable_model == 'cat':
            color = '#9467BD'  # mpl C4 - purple
        # Top axis: Mean Absolute Error
        axes[0].scatter(learn_idx, mae[learn_idx], color=color)
        # Bottom axis: box-plots
        axes[1].boxplot(x=batch_errors_np[:, learn_idx], positions=[learn_idx],
                        widths=0.8, flierprops={'marker': '.', 'markersize': 0.5},
                        boxprops={'color': color})
    axes[0].grid()
    axes[0].set(ylabel='MAE')
    y_max = max(mae.max()*1.02, mae_y_limit)  # dynamic limit
    axes[0].set_ylim([0.0, y_max])
    axes[0].set_title("Synth parameters inference error (blue/purple: numerical/categorical)")
    axes[1].grid(axis='y')
    axes[1].set(ylabel='Inference error')
    axes[1].set_ylim(boxplots_y_limits)
    axes[1].set_xticklabels(_get_learnable_preset_xticklabels(idx_helper))
    _configure_params_plot_x_tick_labels(axes[1])
    # Param groups separations lines
    if len(param_groups_separations) > 0:
        for row in range(len(axes)):
            axes[row].vlines(param_groups_separations, 0.0, 1.0,
                             transform=axes[row].get_xaxis_transform(), colors='C9', linewidth=1.0)
    fig.tight_layout()
    return fig, axes

