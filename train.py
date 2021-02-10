"""
This script performs a single training run for the configuration described
in config.py, when running as __main__.

Its train_config(...) function can also be called from another script,
with small modifications to the config (enqueued train runs).

See train_queue.py for enqueued training runs
"""

import sys
import os
from pathlib import Path
import contextlib
import importlib

import mkl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim
from torch.autograd import profiler

import config
import model.loss
import model.build
import logs.logger
import logs.metrics
from logs.metrics import SimpleMetric, EpochMetric, LatentMetric
import data.dataset
import utils.data
import utils.profile
from utils.hparams import LinearDynamicParam
import utils.figures


def train_config():
    """ Performs a full training run, as described by parameters in config.py.

     Some attributes from config.py might be dynamically changed by train_queue.py - so they
     can be different from what's currently written in config.py. """


    # ========== Logger init (required to load from checkpoint) and Config check ==========
    root_path = Path(__file__).resolve().parent
    logger = logs.logger.RunLogger(root_path, config.model, config.train)
    if logger.restart_from_checkpoint:
        model.build.check_configs_on_resume_from_checkpoint(config.model, config.train,
                                                            logger.get_previous_config_from_json())
    if config.train.start_epoch > 0:  # Resume from checkpoint?
        start_checkpoint = logs.logger.get_model_checkpoint(root_path, config.model, config.train.start_epoch - 1)
    else:
        start_checkpoint = None


    # ========== Datasets and DataLoaders ==========
    full_dataset = data.dataset.DexedDataset(note_duration=config.model.note_duration,
                                             n_fft=config.model.stft_args[0], fft_hop=config.model.stft_args[1],
                                             n_mel_bins=config.model.mel_bins)
    # dataset and dataloader are dicts with 'train', 'validation' and 'test' keys
    dataset = utils.data.random_split(full_dataset, config.train.datasets_proportions, random_gen_seed=0)
    dataloader = dict()
    _debugger = False
    if config.train.profiler_args['enabled'] and config.train.profiler_args['use_cuda']:
        num_workers = 0  # CUDA PyTorch profiler does not work with a multiprocess-dataloader
    elif sys.gettrace() is not None:
        _debugger = True
        print("Debugger detected - num_workers=0 for all DataLoaders")
        num_workers = 0  # PyCharm debug behaves badly with multiprocessing...
    else:  # We should use an higher CPU count for real-time audio rendering
        # 4*GPU count: optimal w/ light dataloader (e.g. (mel-)spectrogram computation)
        num_workers = min(config.train.minibatch_size, torch.cuda.device_count() * 4)
    for dataset_type in dataset:
        # Persistent workers crash with pin_memory - but are more efficient than pinned memory
        dataloader[dataset_type] = DataLoader(dataset[dataset_type], config.train.minibatch_size, shuffle=True,
                                              num_workers=num_workers, pin_memory=False, persistent_workers=True)
        if config.train.verbosity >= 1:
            print("Dataset '{}' contains {}/{} samples ({:.1f}%). num_workers={}".format(dataset_type, len(dataset[dataset_type]), len(full_dataset), 100.0 * len(dataset[dataset_type])/len(full_dataset), num_workers))


    # ========== Model definition ==========
    _, _, ae_model = model.build.build_ae_model(config.model, config.train)
    if start_checkpoint is not None:
        ae_model.load_state_dict(start_checkpoint['ae_model_state_dict'])  # GPU tensor params
    ae_model.eval()
    logger.init_with_model(ae_model, config.model.input_tensor_size)  # model must not be parallel


    # ========== Training devices (GPU(s) only) ==========
    if config.train.verbosity >= 1:
        print("Intel MKL num threads = {}. PyTorch num threads = {}. CUDA devices count: {} GPU(s)."
              .format(mkl.get_max_threads(), torch.get_num_threads(), torch.cuda.device_count()))
    if torch.cuda.device_count() == 0:
        raise NotImplementedError()  # CPU training not available
    elif torch.cuda.device_count() == 1 or config.train.profiler_1_GPU:
        if config.train.profiler_1_GPU:
            print("Using 1/{} GPUs for code profiling".format(torch.cuda.device_count()))
        device = 'cuda:0'
        ae_model = ae_model.to(device)
        ae_model_parallel = nn.DataParallel(ae_model, device_ids=[0])  # "Parallel" 1-GPU model
    else:
        device = torch.device('cuda')
        ae_model.to(device)
        ae_model_parallel = nn.DataParallel(ae_model)  # We use all available GPUs


    # ========== Losses (criterion functions) ==========
    if config.train.ae_reconstruction_loss == 'MSE':
        reconstruction_criterion = nn.MSELoss(reduction='mean')
    else:
        raise NotImplementedError()
    if config.train.latent_loss == 'Dkl':
        latent_criterion = model.loss.GaussianDkl(normalize=config.train.normalize_latent_loss)
    else:
        raise NotImplementedError()


    # ========== Scalars, metrics, images and audio to be tracked in Tensorboard ==========
    scalars = {'ReconsLoss/Train': EpochMetric(), 'ReconsLoss/Valid': EpochMetric(),
               'LatLoss/Train': EpochMetric(), 'LatLoss/Valid': EpochMetric(),
               'VAELoss/Train': SimpleMetric(), 'VAELoss/Valid': SimpleMetric(),
               'ContLoss/Train': EpochMetric(), 'ContLoss/Valid': EpochMetric(),
               'LatCorr/Train': LatentMetric(config.model.dim_z, len(dataset['train'])),
               'LatCorr/Valid': LatentMetric(config.model.dim_z, len(dataset['validation'])),
               'Sched/LR': SimpleMetric(config.train.initial_learning_rate),
               'Sched/beta': LinearDynamicParam(config.train.beta_start_value, config.train.beta,
                                                end_epoch=config.train.beta_warmup_epochs,
                                                current_epoch=config.train.start_epoch)}
    # Losses here are Validation losses. Metrics need an '_' to be different from scalars (tensorboard mixes them)
    metrics = {'ReconsLoss/Valid_': logs.metrics.BufferedMetric(),
               'LatLoss/Valid_': logs.metrics.BufferedMetric(),
               'LatCorr/Valid_': logs.metrics.BufferedMetric(),
               'ContLoss/Valid_': logs.metrics.BufferedMetric(),  # TODO compensated MSE
               'epochs': config.train.start_epoch}
    # TODO check metrics as required in config.py
    logger.tensorboard.init_hparams_and_metrics(metrics)  # hparams added knowing config.*


    # ========== Optimizer and Scheduler ==========
    ae_model.train()
    if config.train.optimizer == 'Adam':
        optimizer = torch.optim.Adam(ae_model.parameters(), lr=config.train.initial_learning_rate,
                                     weight_decay=config.train.weight_decay, betas=config.train.adam_betas)
    else:
        raise NotImplementedError()
    if config.train.scheduler_name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.\
            ReduceLROnPlateau(optimizer, factor=config.train.scheduler_lr_factor, patience=config.train.scheduler_patience,
                              threshold=config.train.scheduler_threshold, verbose=(config.train.verbosity >= 2))
    else:
        raise NotImplementedError()
    if start_checkpoint is not None:
        optimizer.load_state_dict(start_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(start_checkpoint['scheduler_state_dict'])


    # ========== PyTorch Profiling (optional) ==========
    is_profiled = config.train.profiler_args['enabled']
    ae_model.is_profiled = is_profiled


    # ========== Model training epochs ==========
    early_stop = False  # Early stop on final loss plateau
    for epoch in range(config.train.start_epoch, config.train.n_epochs):
        # = = = = = Re-init of epoch metrics = = = = =
        for _, s in scalars.items():
            s.on_new_epoch()
        should_plot = (epoch % config.train.plot_period == 0)

        # = = = = = Train all mini-batches (optional profiling) = = = = =
        # when profiling is disabled: true no-op context manager, and prof is None
        with utils.profile.get_optional_profiler(config.train.profiler_args) as prof:
            ae_model.train()
            dataloader_iter = iter(dataloader['train'])
            for i in range(len(dataloader['train'])):
                with profiler.record_function("DATA_LOAD") if is_profiled else contextlib.nullcontext():
                    sample = next(dataloader_iter)
                    x_in, params_in, sample_info = sample[0].to(device), sample[1].to(device), sample[2].to(device)
                optimizer.zero_grad()
                z_mu_logvar, z_sampled, x_out = ae_model_parallel(x_in)
                scalars['LatCorr/Train'].append(z_mu_logvar, z_sampled)
                with profiler.record_function("BACKPROP") if is_profiled else contextlib.nullcontext():
                    recons_loss = reconstruction_criterion(x_out, x_in)
                    scalars['ReconsLoss/Train'].append(recons_loss)
                    lat_loss = latent_criterion(z_mu_logvar[:, 0, :], z_mu_logvar[:, 1, :])
                    scalars['LatLoss/Train'].append(lat_loss)
                    lat_loss *= scalars['Sched/beta'].get(epoch)
                    scalars['ContLoss/Train'].append(0.0)  # TODO
                    (recons_loss + lat_loss).backward()  # Actual backpropagation is here
                with profiler.record_function("OPTIM_STEP") if is_profiled else contextlib.nullcontext():
                    optimizer.step()  # Internal params. update; before scheduler step
                logger.on_minibatch_finished(i)
                # For full-trace profiling: we need to stop after a few mini-batches
                if config.train.profiler_full_trace and i == 2:
                    break
        if prof is not None:
            logger.save_profiler_results(prof, config.train.profiler_full_trace)
        if config.train.profiler_full_trace:
            break  # Forced training stop
        scalars['VAELoss/Train'] = SimpleMetric(scalars['ReconsLoss/Train'].get() + scalars['LatLoss/Train'].get())

        # = = = = = Evaluation on validation dataset (no profiling) = = = = =
        with torch.no_grad():
            ae_model_parallel.eval()  # BN stops running estimates
            for i, sample in enumerate(dataloader['validation']):
                x_in, params_in, sample_info = sample[0].to(device), sample[1].to(device), sample[2].to(device)
                z_mu_logvar, z_sampled, x_out = ae_model_parallel(x_in)
                scalars['LatCorr/Valid'].append(z_mu_logvar, z_sampled)
                recons_loss = reconstruction_criterion(x_out, x_in)
                scalars['ReconsLoss/Valid'].append(recons_loss)
                lat_loss = latent_criterion(z_mu_logvar[:, 0, :], z_mu_logvar[:, 1, :])
                scalars['LatLoss/Valid'].append(lat_loss)
                lat_loss *= scalars['Sched/beta'].get(epoch)
                scalars['ContLoss/Valid'].append(0.0)  # TODO
                # tensorboard samples for minibatch 'eval' [0] only
                if i == 0 and should_plot:
                    fig, _ = utils.figures.plot_spectrograms(x_in, x_out, sample_info[:, 0], plot_error=True,
                                                             max_nb_specs=config.train.logged_samples_count,
                                                             add_colorbar=True)
                    logger.tensorboard.add_figure('Spectrogram', fig, epoch, close=True)
        scalars['VAELoss/Valid'] = SimpleMetric(scalars['ReconsLoss/Valid'].get() + scalars['LatLoss/Valid'].get())
        # Dynamic LR scheduling depends on validation performance. Loss for plateau-detection is chosen in config.py
        scheduler.step(scalars['{}/Valid'.format(config.train.scheduler_loss)].get())
        scalars['Sched/LR'] = logs.metrics.SimpleMetric(optimizer.param_groups[0]['lr'])
        early_stop = (optimizer.param_groups[0]['lr'] < config.train.early_stop_lr_threshold)  # Early stop?

        # = = = = = Epoch logs (scalars/sounds/images + updated metrics) = = = = =
        for k, s in scalars.items():  # All available scalars are written to tensorboard
            logger.tensorboard.add_scalar(k, s.get(), epoch)
        if should_plot or early_stop:
            fig, _ = utils.figures.plot_latent_distributions_stats(latent_metric=scalars['LatCorr/Valid'])
            logger.tensorboard.add_figure('LatentMu', fig, epoch)
            fig, _ = utils.figures.plot_spearman_correlation(latent_metric=scalars['LatCorr/Valid'])
            logger.tensorboard.add_figure('LatentEntanglement', fig, epoch)
        metrics['epochs'] = epoch + 1
        metrics['ReconsLoss/Valid_'].append(scalars['ReconsLoss/Valid'].get())
        metrics['LatLoss/Valid_'].append(scalars['LatLoss/Valid'].get())
        metrics['LatCorr/Valid_'].append(scalars['LatCorr/Valid'].get())
        metrics['ContLoss/Valid_'].append(scalars['ContLoss/Valid'].get())
        logger.tensorboard.update_metrics(metrics)

        # = = = = = Model+optimizer(+scheduler) save - ready for next epoch = = = = =
        if (epoch % config.train.save_period == 0) or (epoch == config.train.n_epochs-1) or early_stop:
            logger.save_checkpoint(epoch, ae_model, optimizer, scheduler)
        logger.on_epoch_finished(epoch)
        if early_stop:
            print("[train.py] Training stopped early (final loss plateau)")
            break


    # ========== Logger final stats ==========
    logger.on_training_finished()


if __name__ == "__main__":
    # Normal run, config.py only will be used to parametrize learning and models
    train_config()

