"""
Performs training for the configuration described in config.py
"""

import os
from pathlib import Path
import contextlib

import mkl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim
from torch.autograd import profiler

import config
from model import VAE, encoder, decoder
import log.logger
import data.dataset
import utils.data
import utils.profile


# ========== Datasets and DataLoaders ==========
full_dataset = data.dataset.DexedDataset(note_duration=config.model.note_duration,
                                         n_fft=config.model.stft_args[0], fft_hop=config.model.stft_args[1])
# dataset and dataloader are dicts with 'train', 'validation' and 'test' keys
dataset = utils.data.random_split(full_dataset, config.train.datasets_proportions, random_gen_seed=0)
dataloader = dict()
if config.train.profiler_args['enabled'] and config.train.profiler_args['use_cuda']:
    num_workers = 0  # CUDA PyTorch profiler does not work with a multiprocess-dataloader
else:
    num_workers = min(config.train.minibatch_size, os.cpu_count()*3//4)
for dataset_type in dataset:
    dataloader[dataset_type] = DataLoader(dataset[dataset_type], config.train.minibatch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=True)
    if config.train.verbosity >= 1:
        print("Dataset '{}' contains {}/{} samples ({:.1f}%). num_workers={}".format(dataset_type, len(dataset[dataset_type]), len(full_dataset), 100.0 * len(dataset[dataset_type])/len(full_dataset), num_workers))


# ========== Model definition ==========
# Encoder and decoder with the same architecture
encoder_model = encoder.SpectrogramEncoder(config.model.encoder_architecture, config.model.dim_z,
                                           config.model.spectrogram_size)
decoder_model = decoder.SpectrogramDecoder(config.model.encoder_architecture, config.model.dim_z,
                                           config.model.spectrogram_size)
ae_model = VAE.BasicVAE(encoder_model, config.model.dim_z, decoder_model)  # Not parallelized yet


# ========== Logger init ==========
ae_model.eval()
logger = log.logger.RunLogger(Path(__file__).resolve().parent, config.model, config.train)
logger.init_with_model(ae_model, config.model.input_tensor_size)  # model must not be parallel


# ========== Training devices ==========
if config.train.verbosity >= 1:
    print("Intel MKL num threads = {}".format(mkl.get_max_threads()))
    print("PyTorch num threads = {}".format(torch.get_num_threads()))
    print("CUDA devices count: {} GPU(s)".format(torch.cuda.device_count()))
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


# ========== Losses and Metrics ==========
if config.train.ae_reconstruction_loss == 'MSE':
    loss = nn.MSELoss()
else:
    raise NotImplementedError()
metrics = {'dummy_metric': 0.07}  # TODO
logger.tensorboard.init_hparams_and_metrics(metrics)


# ========== Optimizer and Scheduler ==========
ae_model.train()
optimizer = torch.optim.Adam(ae_model.parameters())


# ========== PyTorch Profiling (optional) ==========
is_profiled = config.train.profiler_args['enabled']
ae_model.is_profiled = is_profiled


# ========== Model training epochs ==========
# TODO consider start epoch
for epoch in range(0, config.train.n_epochs):
    # = = = = = Train all mini-batches (optional profiling) = = = = =
    with utils.profile.get_optional_profiler(config.train.profiler_args) as prof:  # if disabled: true no-op and prof is None
        dataloader_iter = iter(dataloader['train'])
        for i in range(len(dataloader['train'])):
            with profiler.record_function("DATA_LOAD") if is_profiled else contextlib.nullcontext():
                sample = next(dataloader_iter)
                x_in, params_in, midi_in = sample[0].to(device), sample[1].to(device), sample[2].to(device)
            optimizer.zero_grad()
            x_out = ae_model_parallel(x_in)
            with profiler.record_function("BACKPROP") if is_profiled else contextlib.nullcontext():
                l = loss(x_out, x_in)
                l.backward()
            with profiler.record_function("OPTIM_STEP") if is_profiled else contextlib.nullcontext():
                optimizer.step()  # TODO refaire propre
            print("epoch {} batch {}".format(epoch, i))
            # For full-trace profiling: we need to stop after a few mini-batches
            if config.train.profiler_full_trace and i == 2:
                break
    if prof is not None:
        logger.save_profiler_results(prof)
    if config.train.profiler_full_trace:
        break  # Forced training stop

    # TODO = = = = = Evaluation on validation dataset = = = = =
    # TODO  = = = = = Epoch logs (scalars/sounds/images + updated metrics) = = = = =
    logger.tensorboard.update_metrics(metrics)
    # TODO  = = = = = Model save - ready for next epoch = = = = =
    logger.on_epoch_finished(epoch, ae_model)


# ========== Logger final stats ==========
logger.on_training_finished()

