"""
Script that can be edited to configure and run a queue of training runs.
Must be run as main

See the actual training function in train.py
"""


import importlib  # to reload config.py between each run
import numpy as np
import copy
import time

import torch

import train
import utils.exception


# TODO intercept Ctrl-C sigint and ask for confirmation



model_config_mods, train_config_mods = list(), list()
"""
Please write two lists of dicts, such that:
- (model|train)_config_mods contains the modifications applied to config.model and config.train, resp.
- each list index corresponds to a training run
- each dict key corresponds to an attribute of config.model.* or config.train.*. Empty dict to indicate
      that no config modification should be performed
"""

# automatically train all cross-validation folds?
train_all_k_folds = True


# Run 0 - example
model_config_mods.append({'name': 'FlVAE3',
                          'run_name': '15b_dex6op_all<=32_1midi',
                          'synth_vst_params_learned_as_categorical': 'all<=32'})
train_config_mods.append({'main_cuda_device_idx': 1})





if __name__ == "__main__":
    assert len(model_config_mods) == len(train_config_mods)

    import config  # Will be reloaded on each new run

    # If performing k-fold cross validation trains, duplicate run mods to train all folds
    if train_all_k_folds:
        model_config_mods_kfolds, train_config_mods_kfolds = list(), list()
        for base_run_index in range(len(model_config_mods)):
            for fold_idx in range(config.train.k_folds):
                # duplicate this run configuration, one duplicate per fold
                model_config_mods_kfolds.append(copy.deepcopy(model_config_mods[base_run_index]))
                train_config_mods_kfolds.append(copy.deepcopy(train_config_mods[base_run_index]))
                train_config_mods_kfolds[-1]['current_k_fold'] = fold_idx
                # k-fold index appended to the name
                if 'run_name' in model_config_mods[base_run_index]:
                    run_name = model_config_mods[base_run_index]['run_name'] + '_kf{}'.format(fold_idx)
                else:
                    run_name = config.model.run_name + '_kf{}'.format(fold_idx)
                model_config_mods_kfolds[-1]['run_name'] = run_name
        model_config_mods, train_config_mods = model_config_mods_kfolds, train_config_mods_kfolds


    # = = = = = = = = = = Training queue: main loop = = = = = = = = = =
    for run_index in range(len(model_config_mods)):
        # Force config reload
        importlib.reload(config)

        print("================================================================")
        print("=============== Enqueued Training Run {}/{} starts ==============="
              .format(run_index+1, len(model_config_mods)))

        # Direct dirty modification of config.py module attributes
        # The dynamically modified config.py will be used by train.py
        for k, v in model_config_mods[run_index].items():
            config.model.__dict__[k] = v
        for k, v in train_config_mods[run_index].items():
            config.train.__dict__[k] = v
        config.update_dynamic_config_params()  # Required if we modified critical hyper-parameters

        # Model train. An occasional model divergence (sometimes happen during first epochs) is tolerated
        #    Full-AR Normalizing Flows (e.g. MAF/IAF) are very unstable and hard to train on dim>100 latent spaces
        max_divergent_model_runs = 2  # 2 diverging runs are already a lot... a 3rd diverging run stops training
        divergent_model_runs = 0
        has_finished_training = False
        while not has_finished_training:
            try:  # - - - - - Model train - - - - -
                train.train_config()
                has_finished_training = True
            except utils.exception.ModelConvergenceError as e:
                divergent_model_runs += 1
                if divergent_model_runs <= max_divergent_model_runs:
                    print("[train_queue.py] Model train did not converge: {}. Restarting run... (next trial: {}/{})"
                          .format(e, divergent_model_runs + 1, max_divergent_model_runs + 1))
                    config.model.allow_erase_run = True  # We force the run to be erasable
                else:
                    e_str = "Model training run {}/{} does not converge ({} run trials failed). " \
                            "Training queue will now stop, please check this convergence problem."\
                        .format(run_index+1, len(model_config_mods), divergent_model_runs)
                    raise utils.exception.ModelConvergenceError(e_str)

        print("=============== Enqueued Training Run {}/{} has finished ==============="
              .format(run_index+1, len(model_config_mods)))
        print("======================================================================")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Maybe PyTorch / Python GC need some time to empty CUDA buffers...
        # An out-of-memory crash remains unexplained
        time.sleep(20)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

