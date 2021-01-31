
import os
import time
import shutil
import json
import datetime

import numpy as np
import torch

import humanize
import torchinfo

from .tbwriter import TensorboardSummaryWriter  # Custom modified summary writer

_erase_security_time_s = 2.0


def get_model_run_directory(root_path, model_config):
    """ Returns the directory where saved models and config.json are stored, for a particular run.
    Does not check whether the directory exists or not (it must have been created by the RunLogger) """
    return root_path.joinpath(model_config.logs_root_dir)\
        .joinpath(model_config.name).joinpath(model_config.run_name)


def get_tensorboard_run_directory(root_path, model_config):
    """ Returns the directory where Tensorboard model metrics are stored, for a particular run. """
    # pb s'il y en a plusieurs ? (semble résolu avec override de add_hparam PyTorch)
    return root_path.joinpath(model_config.logs_root_dir).joinpath('runs')\
        .joinpath(model_config.name).joinpath(model_config.run_name)


def erase_run_data(root_path, model_config):
    """ Erases all previous data (Tensorboard, config, saved models)
    for a particular run of the model. """
    if _erase_security_time_s > 0.1:
        print("[RunLogger] *** WARNING *** '{}' run for model '{}' will be erased in {} seconds. "
              "Stop this program to cancel ***"
              .format(model_config.run_name, model_config.name, _erase_security_time_s))
        time.sleep(_erase_security_time_s)
    else:
        print("[RunLogger] '{}' run for model '{}' will be erased.".format(model_config.run_name, model_config.name))
    shutil.rmtree(get_model_run_directory(root_path, model_config))  # config and saved models
    shutil.rmtree(get_tensorboard_run_directory(root_path, model_config))  # tensorboard


class RunLogger:
    """ Class for saving interesting data during a training run:
     - graphs, losses, metrics, and some results to Tensorboard
     - config.py as a json file
     - trained models

     See ../README.md to get more info on storage location.

     TODO do not create a new run if training re-starts from epoch > 0
     """
    def __init__(self, root_path, model_config, train_config, minibatches_count=0):
        """

        :param root_path: pathlib.Path of the project's root folder
        :param model_config: from config.py
        :param train_config: from config.py
        :param minibatches_count: Length of the 'train' dataloader
        """
        # Configs are stored but not modified by this class
        self.model_config = model_config
        self.train_config = train_config
        self.verbosity = train_config.verbosity
        global _erase_security_time_s  # Very dirty.... but quick
        _erase_security_time_s = train_config.init_security_pause
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
        # - - - - - Epochs, Batches, ... - - - - -
        self.minibatches_count = minibatches_count
        self.minibatch_duration_running_avg = 0.0
        self.minibatch_duration_avg_coeff = 0.05  # auto-regressive running average coefficient
        self.last_minibatch_start_datetime = datetime.datetime.now()
        self.epoch_start_datetimes = [datetime.datetime.now()]  # This value can be erased in init_with_model
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
        self.epoch_start_datetimes = [datetime.datetime.now()]

    def on_minibatch_finished(self, minibatch_idx):
        # TODO time stats - running average
        minibatch_end_time = datetime.datetime.now()
        delta_t = (minibatch_end_time - self.last_minibatch_start_datetime).total_seconds()
        self.minibatch_duration_running_avg *= (1.0 - self.minibatch_duration_avg_coeff)
        self.minibatch_duration_running_avg += self.minibatch_duration_avg_coeff * delta_t
        if self.verbosity >= 3:
            print("epoch {} batch {} delta t = {}ms" .format(len(self.epoch_start_datetimes)-1, minibatch_idx,
                                                             int(1000.0 * self.minibatch_duration_running_avg)))
        self.last_minibatch_start_datetime = minibatch_end_time

    def on_epoch_finished(self, epoch, model_to_save):
        # TODO add args and implement...
        self.epoch_start_datetimes.append(datetime.datetime.now())
        # TODO save model
        epoch_duration = self.epoch_start_datetimes[-1] - self.epoch_start_datetimes[-2]
        avg_duration_s = np.asarray([(self.epoch_start_datetimes[i+1] - self.epoch_start_datetimes[i]).total_seconds()
                                     for i in range(len(self.epoch_start_datetimes) - 1)])
        avg_duration_s = avg_duration_s.mean()
        remaining_datetime = avg_duration_s * (self.train_config.n_epochs - (epoch-self.train_config.start_epoch) - 1)
        remaining_datetime = datetime.timedelta(seconds=int(remaining_datetime))
        if self.verbosity >= 1:
            print("End of epoch {} ({}/{}). Duration={:.1f}s, avg={:.1f}s. Estimated remaining time: {} ({})"
                  .format(epoch, epoch-self.train_config.start_epoch+1, self.train_config.n_epochs,
                          epoch_duration.total_seconds(), avg_duration_s,
                          remaining_datetime, humanize.naturaldelta(remaining_datetime)))

    def save_profiler_results(self, prof, save_full_trace=False):
        """ Saves (overwrites) current profiling results.
        Warning: do not save full trace for long learning (approx. 10MB per **mini_batch**) """
        # TODO Write several .txt files with different sort methods
        with open(self.run_dir.joinpath('profiling_by_cuda_time.txt'), 'w') as f:
            f.write(prof.key_averages(group_by_stack_n=5).table(sort_by='cuda_time_total').__str__())
        if save_full_trace:
            prof.export_chrome_trace(self.run_dir.joinpath('profiling_chrome_trace.json'))

    def on_training_finished(self):
        # TODO write training stats
        self.tensorboard.flush()
        self.tensorboard.close()
        if self.train_config.verbosity >= 1:
            print("[RunLogger] Training has finished")


