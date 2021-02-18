"""
Utility function for building datasets and dataloaders using given configuration arguments.
"""

import sys
import numpy as np

import torch.utils.data
from torch.utils.data import DataLoader

from . import dataset


def random_split(full_dataset, datasets_proportions, random_gen_seed=0):
    """ Wrapper for torch.utils.data.random_split which takes 3 float proportions instead
     of integer lengths as input.

     Returns a dict of 3 sub-datasets with keys 'train', 'validation' and 'test'. """
    # TODO arg to choose the current k-fold
    # TODO test set must *always* remain the same, for a given full_dataset
    #    (use a random split first (always the same seed) then a deterministic split for k-fold?)
    assert len(datasets_proportions) == 3
    sub_dataset_lengths = [int(np.floor(r * len(full_dataset))) for r in datasets_proportions]
    sub_dataset_lengths[2] = len(full_dataset) - sub_dataset_lengths[0] - sub_dataset_lengths[1]
    # Indexes are randomly split, but the gen seed must always be the same
    sub_datasets = torch.utils.data.random_split(full_dataset, sub_dataset_lengths,
                                                 generator=torch.Generator().manual_seed(random_gen_seed))
    return {'train': sub_datasets[0], 'validation': sub_datasets[1], 'test': sub_datasets[2]}


def get_full_and_split_datasets(model_config, train_config):
    if model_config.synth.startswith('dexed'):
        full_dataset = dataset.DexedDataset(** dataset.model_config_to_dataset_kwargs(model_config),
                                            algos=model_config.dataset_synth_args[0],
                                            operators=model_config.dataset_synth_args[1],
                                            restrict_to_labels=model_config.dataset_labels)
    else:
        raise NotImplementedError("No dataset available for '{}': unrecognized synth.".format(model_config.synth))
    # dataset and dataloader are dicts with 'train', 'validation' and 'test' keys
    # TODO "test" holdout dataset must *always* be the same - even when performing k-fold cross-validation
    #   do 2 random splits?
    sub_datasets = random_split(full_dataset, train_config.datasets_proportions, random_gen_seed=0)
    return full_dataset, sub_datasets


def get_split_dataloaders(train_config, full_dataset, sub_datasets, persistent_workers=True):
    sub_dataloaders = dict()
    _debugger = False
    if train_config.profiler_args['enabled'] and train_config.profiler_args['use_cuda']:
        num_workers = 0  # CUDA PyTorch profiler does not work with a multiprocess-dataloader
    elif sys.gettrace() is not None:
        _debugger = True
        print("[data/build.py] Debugger detected - num_workers=0 for all DataLoaders")
        num_workers = 0  # PyCharm debug behaves badly with multiprocessing...
    else:  # We should use an higher CPU count for real-time audio rendering
        # 4*GPU count: optimal w/ light dataloader (e.g. (mel-)spectrogram computation)
        num_workers = min(train_config.minibatch_size, torch.cuda.device_count() * 4)
    for dataset_type in sub_datasets:
        # Persistent workers crash with pin_memory - but are more efficient than pinned memory
        sub_dataloaders[dataset_type] = DataLoader(sub_datasets[dataset_type], train_config.minibatch_size,
                                                   shuffle=True, num_workers=num_workers, pin_memory=False,
                                                   persistent_workers=((num_workers > 0) and persistent_workers))
        if train_config.verbosity >= 1:
            print("[data/build.py] Dataset '{}' contains {}/{} samples ({:.1f}%). num_workers={}"
                  .format(dataset_type, len(sub_datasets[dataset_type]), len(full_dataset),
                          100.0 * len(sub_datasets[dataset_type])/len(full_dataset), num_workers))
    return sub_dataloaders

