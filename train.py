"""
Performs training for the configuration described in config.py
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim
import mkl

import config
from model import VAE, encoder, decoder
import log.logger
import data.dataset
import utils.data
from torch.autograd import profiler


print("MKL num threads = {}".format(mkl.get_max_threads()))
print("PyTorch num threads = {}".format(torch.get_num_threads()))


# ========== Datasets and DataLoaders ==========
full_dataset = data.dataset.DexedDataset(note_duration=config.model.note_duration,
                                         n_fft=config.model.stft_args[0], fft_hop=config.model.stft_args[1])
# dataset and dataloader are dicts with 'train', 'validation' and 'test' keys
dataset = utils.data.random_split(full_dataset, config.train.datasets_proportions, random_gen_seed=0)
dataloader = dict()
if config.train.profiler_args['enabled'] and config.train.profiler_args['use_cuda']:
    num_workers = 0  # CUDA PyTorch profiler does not work with a multiprocess-dataloader
else:
    num_workers = min(config.train.minibatch_size, os.cpu_count()//2)
for dataset_type in dataset:
    dataloader[dataset_type] = DataLoader(dataset[dataset_type], config.train.minibatch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=True)


# ========== Model definition ==========
# Encoder and decoder with the same architecture
encoder_model = encoder.SpectrogramEncoder(config.model.encoder_architecture, config.model.dim_z,
                                           config.model.spectrogram_size)
decoder_model = decoder.SpectrogramDecoder(config.model.encoder_architecture, config.model.dim_z,
                                           config.model.spectrogram_size)
ae_model = VAE.BasicVAE(encoder_model, config.model.dim_z, decoder_model)


# ========== Losses and Metrics ==========
if config.train.ae_reconstruction_loss == 'MSE':
    loss = nn.MSELoss()
else:
    raise NotImplementedError()
metrics = {'dummy_metric': 0.07}  # TODO


# ========== Logger init ==========
ae_model.eval()
logger = log.logger.RunLogger(Path(__file__).resolve().parent, config.model, config.train)
logger.init_with_model(ae_model, config.model.input_tensor_size)
logger.tensorboard.init_hparams_and_metrics(metrics)


# ========== Optimizer and Scheduler ==========
device = torch.device(config.train.device)
ae_model = ae_model.to(device)
ae_model.train()
optimizer = torch.optim.Adam(ae_model.parameters())


# ========== PyTorch Profiling (optional) ==========
if config.train.profiler_args['enabled']:
    pass # TODO structure propre
else:
    raise NotImplementedError()


# ========== Model training epochs ==========
# TODO consider start epoch
for epoch in range(0, config.train.n_epochs):
    with profiler.profile(**config.train.profiler_args) as prof:
        # TODO train all mini-batches
        for i, sample in enumerate(dataloader['train']):
            optimizer.zero_grad()
            x_in, params_in, midi_in = sample[0].to(device), sample[1].to(device), sample[2].to(device)
            #x_out = ae_model(x_in)
            x_out = ae_model.profile_forward(x_in)
            with profiler.record_function("BACKPROP"):
                l = loss(x_out, x_in)
                l.backward()
            with profiler.record_function("OPTIM_STEP"):
                optimizer.step()  # TODO refaire propre
            print("epoch {} batch {}".format(epoch, i))
            if i == 2:  # TODO faire vraiment ça si profile_only
                break
    logger.save_profiler_results(prof)
    break  # TODO nice conditional break
    # TODO evaluation on validation dataset

    # TODO epoch logs (epoch scalars/sounds/images + metrics update)
    logger.tensorboard.update_metrics(metrics)
    logger.on_epoch_finished(epoch, ae_model)


# ========== Logger final stats ==========
logger.on_training_finished()

