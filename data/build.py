"""
Utility function for building datasets and dataloaders using given configuration arguments.
"""

import sys
import numpy as np

import torch.utils.data
from torch.utils.data import DataLoader

from . import dataset
import data.sampler


def get_dataset(model_config, train_config):
    """
    Returns the full (main) dataset.
    If a Flow-based synth params regression is to be used, this function will modify the latent space
    dimension dim_z on the config.py module directly (its model attribute is given as an arg of this function).
    """
    if model_config.synth.startswith('dexed'):
        full_dataset = dataset.DexedDataset(** dataset.model_config_to_dataset_kwargs(model_config),
                                            algos=model_config.dataset_synth_args[0],
                                            operators=model_config.dataset_synth_args[1],
                                            vst_params_learned_as_categorical=
                                            model_config.synth_vst_params_learned_as_categorical,
                                            restrict_to_labels=model_config.dataset_labels)
    else:
        raise NotImplementedError("No dataset available for '{}': unrecognized synth.".format(model_config.synth))
    if train_config.verbosity >= 2:
        print(full_dataset.preset_indexes_helper)
    elif train_config.verbosity >= 1:
        print(full_dataset.preset_indexes_helper.short_description)
    # config.py direct dirty modifications - number of learnable params depends on the synth and dataset arguments
    model_config.synth_params_count = full_dataset.learnable_params_count
    model_config.learnable_params_tensor_length = full_dataset.learnable_params_tensor_length
    if model_config.params_regression_architecture.startswith("flow_"):
        # ********************************* dim_z changes if a flow network is used *********************************
        model_config.dim_z = model_config.learnable_params_tensor_length
    return full_dataset


def get_split_dataloaders(train_config, full_dataset, persistent_workers=True):
    """ Returns a dict of train/validation/test DataLoader instances, and a dict which contains the
    length of each sub-dataset. """
    # Num workers might be zero (no multiprocessing)
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
    # Dataloader easily build from samplers
    subset_samplers = data.sampler.build_subset_samplers(full_dataset, k_fold=train_config.current_k_fold,
                                                         k_folds_count=train_config.k_folds,
                                                         test_holdout_proportion=train_config.test_holdout_proportion)
    dataloaders = dict()
    sub_datasets_lengths = dict()
    for k, sampler in subset_samplers.items():
        # Last train minibatch must be dropped to help prevent training instability. Worst case example, last minibatch
        # contains only 8 elements, mostly sfx: these hard to learn (or generate) item would have a much higher
        # equivalent learning rate because all losses are minibatch-size normalized. No issue for eval though
        drop_last = (k.lower() == 'train')
        # Dataloaders based on previously built samplers
        dataloaders[k] = torch.utils.data.DataLoader(full_dataset, batch_size=train_config.minibatch_size,
                                                     sampler=sampler, num_workers=num_workers, pin_memory=False,
                                                     persistent_workers=((num_workers > 0) and persistent_workers),
                                                     drop_last=drop_last)
        sub_datasets_lengths[k] = len(sampler.indices)
        if train_config.verbosity >= 1:
            print("[data/build.py] Dataset '{}' contains {}/{} samples ({:.1f}%). num_workers={}"
                  .format(k, sub_datasets_lengths[k], len(full_dataset),
                          100.0 * sub_datasets_lengths[k]/len(full_dataset), num_workers))
    return dataloaders, sub_datasets_lengths

