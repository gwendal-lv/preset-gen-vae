"""
Samplers for any abstract PresetDataset class, which can be used as train/valid/test samplers.
Support k-fold cross validation and subtleties of multi-note (multi-layer spectrogram) preset datasets.

"""

from collections.abc import Iterable
from typing import Dict

import numpy as np
import torch
import torch.utils.data

from data.abstractbasedataset import PresetDataset


def build_subset_samplers(dataset: PresetDataset,
                          k_fold=0, k_folds_count=5, test_holdout_proportion=0.2,
                          random_seed=0
                          ) -> Dict[str, torch.utils.data.SubsetRandomSampler]:
    """
    Builds 'train', 'validation' and 'test' subset samplers

    :param dataset: Required to properly separate dataset items indexes by preset UIDs (not to split
        a multi-note preset into multiple subsets).
    :param k_fold: Current k-fold cross-validation fold index
    :param k_folds_count: Total number of k-folds
    :param test_holdout_proportion: Proportion of 'test' data, excluded from cross-validation folds.
    :param random_seed: For reproducibility, always use the same seed

    :returns: dict of subset_samplers
    """
    presets_count = dataset.valid_presets_count
    all_preset_indexes = np.arange(presets_count)
    preset_indexes = dict()
    rng = np.random.default_rng(seed=random_seed)
    # Shuffle preset indexes, and separate them into subsets
    rng.shuffle(all_preset_indexes)  # in-place shuffling
    first_test_idx = int(np.floor(presets_count * (1.0 - test_holdout_proportion)))
    non_test_preset_indexes, preset_indexes['test'] = np.split(all_preset_indexes, [first_test_idx])
    # All folds are retrieved - we'll choose only one of these as validation subset, and merge the others
    preset_indexes_folds = np.array_split(non_test_preset_indexes, k_folds_count)
    preset_indexes['validation'] = preset_indexes_folds[k_fold]
    preset_indexes['train'] = np.hstack([preset_indexes_folds[i] for i in range(k_folds_count) if i != k_fold])
    # Final indexes
    if dataset.midi_notes_per_preset == 1 or dataset.multichannel_stacked_spectrograms:
        final_indexes = preset_indexes
    else:  # multi-note, single-layer spectrogram dataset: dataset indexes are not preset indexes
        final_indexes = dict()
        # We don't need to shuffle again these groups (SubsetRandomSampler will do it)
        for k in preset_indexes:  # k: train, valid or test
            final_indexes[k] = list()
            for preset_idx in preset_indexes[k]:
                final_indexes[k] += [preset_idx * dataset.midi_notes_per_preset + i
                                     for i in range(dataset.midi_notes_per_preset)]
    subset_samplers = dict()
    for k in final_indexes:
        subset_samplers[k] = torch.utils.data.SubsetRandomSampler(final_indexes[k])
    return subset_samplers


if __name__ == "__main__":
    # Tests on a reduced dexed dataset (1 under-represented algorithm)`
    import pathlib
    import sys
    sys.path.append(pathlib.Path(__file__).parent.parent)
    import config  # Dirty path trick to import config.py from project root dir
    from data.dexeddataset import DexedDataset

    full_dataset =  DexedDataset(note_duration=config.model.note_duration,
                                 midi_notes=((60, 100), (70, 100)),  # config.model.midi_notes,
                                 n_fft=config.model.stft_args[0], fft_hop=config.model.stft_args[1],
                                 algos=[21],  # very few samples
                                 multichannel_stacked_spectrograms=False,  # Set False to test the annoying case
                                 n_mel_bins=config.model.mel_bins,
                                 # Params learned as categorical: maybe comment
                                 vst_params_learned_as_categorical=config.model.synth_vst_params_learned_as_categorical,
                                 spectrogram_min_dB=config.model.spectrogram_min_dB)
    print(full_dataset)

    subset_samplers = build_subset_samplers(full_dataset, k_fold=0)
    dataloaders = dict()
    for k, sampler in subset_samplers.items():
        dataloaders[k] = torch.utils.data.DataLoader(full_dataset, batch_size=1, sampler=sampler)
    # test: load a full dataset and train/valid/test subsets,
    #   and check that a single preset UID cannot be found in 2 subsets
    #   This test will be very long on medium and large datasets

    def test_preset_UIDs_across_subsets(ref_key, other_key1, other_key2):
        for i, batch_ref in enumerate(dataloaders[ref_key]):
            preset_UID_ref = batch_ref[2][0, 0].item()
            for j, batch_other in enumerate(dataloaders[other_key1]):
                assert batch_other[2][0, 0].item() != preset_UID_ref
            for j, batch_other in enumerate(dataloaders[other_key2]):
                assert batch_other[2][0, 0].item() != preset_UID_ref

    test_preset_UIDs_across_subsets('test', 'train', 'validation')
    test_preset_UIDs_across_subsets('validation', 'train', 'test')

    print("Random Samplers: OK, no preset UID dispersion across train/validation/test subsets.")
