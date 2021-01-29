
import os
import time
import shutil
import json
import datetime

import numpy as np
import torch

import torchinfo

from .tbwriter import TensorboardSummaryWriter  # Custom modified summary writer

_erase_security_time_s = 2


def get_model_run_directory(root_path, model_config):
    """ Returns the directory where saved models and config.json are stored, for a particular run.
    Does not check whether the directory exists or not (it must have been created by the RunLogger) """
    return root_path.joinpath(model_config.logs_root_dir)\
        .joinpath(model_config.name).joinpath(model_config.run_name)


def get_tensorboard_run_directory(root_path, model_config):
    """ Returns the directory where Tensorboard model metrics are stored, for a particular run. """
    # TODO gérer pb s'il y en a plusieurs... (pb semble résolu avec màj PyTorch)
    return root_path.joinpath(model_config.logs_root_dir).joinpath('runs')\
        .joinpath(model_config.name).joinpath(model_config.run_name)


def erase_run_data(root_path, model_config):
    """ Erases all previous data (Tensorboard, config, saved models)
    for a particular run of the model. """
    print("[RunLogger] *** WARNING *** '{}' run for model '{}' will be erased in {} seconds. "
          "Stop this program to cancel ***"
          .format(model_config.run_name, model_config.name, _erase_security_time_s))
    time.sleep(_erase_security_time_s)
    shutil.rmtree(get_model_run_directory(root_path, model_config))  # config and saved models
    shutil.rmtree(get_tensorboard_run_directory(root_path, model_config))  # tensorboard


class RunLogger:
    """ Class for saving interesting data during a training run:
     - graphs, losses, metrics, and some results to Tensorboard
     - config.py as a json file
     - trained models

     See ../README.md to get more info on storage location.

     TODO does not create a new run if training re-starts from epoch > 0
     TODO prompt alert if a previous run data is to be overwritten
     TODO deletion of previous run data - if needed
     """
    def __init__(self, root_path, model_config, train_config):
        # Configs are stored but not modified by this class
        self.model_config = model_config
        self.train_config = train_config
        # - - - - - Directories creation (if not exists) for model (not yet for a given run) - - - - -
        self.log_dir = root_path.joinpath(model_config.logs_root_dir).joinpath(model_config.name)
        self._make_dirs_if_dont_exist(self.log_dir)
        self.tensorboard_model_dir = root_path.joinpath(model_config.logs_root_dir)\
            .joinpath('runs').joinpath(model_config.name)
        self._make_dirs_if_dont_exist(self.tensorboard_model_dir)
        # - - - - - Run directories and data management - - - - -
        if self.train_config.verbosity >= 1:
            print("[RunLogger] Starting logging into '{}'".format(self.log_dir))
        self.run_dir = self.log_dir.joinpath(model_config.run_name)  # This is the run's reference folder
        self.saved_models_dir = self.run_dir.joinpath('models')
        self.tensorboard_run_dir = self.tensorboard_model_dir.joinpath(model_config.run_name)
        # Check: does the run folder already exist?
        if not os.path.exists(self.run_dir):
            if train_config.start_epoch != 0:
                raise RuntimeError("config.py error: this new run must start from epoch 0")
            # TODO security: try to erase the corresponding tensorboard run dir
            self._make_model_run_dirs()
            if self.train_config.verbosity >= 1:
                print("[RunLogger] Created '{}' directory to store config and models.".format(self.run_dir))
        # If run folder already exists
        else:
            if not model_config.allow_erase_run:
                raise RuntimeError("Config does not allow to erase the '{}' run for model '{}'"
                                   .format(model_config.run_name, model_config.name))
            else:
                if train_config.start_epoch == 0:  # Start a new fresh training
                    erase_run_data(root_path, model_config)  # module function
                    self._make_model_run_dirs()
                else:
                    raise NotImplementedError("Must load a previous training epoch: not implemented")
        # Write config file on startup only - any previous config file will be erased
        config_dict = {'model': model_config.__dict__, 'train': train_config.__dict__}
        with open(self.run_dir.joinpath('config.json'), 'w') as f:
            json.dump(config_dict, f)
        # Various logged data
        self.epoch_start_datetimes = [datetime.datetime.now()]
        # - - - - - Tensorboard - - - - -
        self.tensorboard = TensorboardSummaryWriter(log_dir=self.tensorboard_run_dir, flush_secs=5,
                                                    model_config=model_config, train_config=train_config)

    @staticmethod
    def _make_dirs_if_dont_exist(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _make_model_run_dirs(self):
        """ Creates (no check) the directories for storing config and saved models. """
        os.makedirs(self.run_dir)
        os.makedirs(self.saved_models_dir)

    def init_with_model(self, model, input_tensor_size):
        """ Finishes to initialize this logger given the fully-build model """
        # TODO consider several models
        description = torchinfo.summary(model, input_size=input_tensor_size, depth=5, device='cpu', verbose=0)
        with open(self.run_dir.joinpath('torchinfo_summary.txt'), 'w') as f:
            f.write(description.__str__())
        self.tensorboard.add_graph(model, torch.zeros(input_tensor_size))

    def on_epoch_finished(self, epoch, model_to_save):
        # TODO add args and implement...
        self.epoch_start_datetimes.append(datetime.datetime.now())
        # TODO move loss to specific function called directly from train.py
        self.tensorboard.add_scalar("MSELoss/dummy", 1 / (1 + epoch*np.random.normal(1.2, 0.1)), epoch)
        # TODO save model
        epoch_duration = self.epoch_start_datetimes[-1] - self.epoch_start_datetimes[-2]
        cout_str = "End of epoch {} (duration: {})".format(epoch, epoch_duration)
        if self.train_config.verbosity == 2:
            print(cout_str)

    def save_profiler_results(self, prof):
        """ Saves (overwrites) current profiling results. """
        # TODO Write several .txt files with different sort methods
        with open(self.run_dir.joinpath('profiling_by_cuda_time.txt'), 'w') as f:
            f.write(prof.key_averages(group_by_stack_n=5).table(sort_by='cuda_time_total').__str__())
        prof.export_chrome_trace(self.run_dir.joinpath('profiling_chrome_trace.json'))

    def on_training_finished(self):
        # TODO write training stats
        self.tensorboard.flush()
        self.tensorboard.close()
        if self.train_config.verbosity >= 1:
            print("[RunLogger] Training has finished")


