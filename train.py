"""
Performs training for the configuration described in config.py
"""

from pathlib import Path

import config
from model import VAE, encoder, decoder
import log.logger


# ========== Datasets and DataLoaders ==========
# TODO - construction longue, on fera en dernier


# ========== Model definition ==========
# Encoder and decoder with the same architecture
encoder_model = encoder.SpectrogramEncoder(config.model.encoder_architecture, config.model.dim_z,
                                           config.model.spectrogram_size)
decoder_model = decoder.SpectrogramDecoder(config.model.encoder_architecture, config.model.dim_z)
ae_model = VAE.BasicVAE(encoder_model, config.model.dim_z, decoder_model)


# ========== Loss and Metrics ==========
metrics = {'dummy_metric': 0.07}  # TODO


# ========== Logger init ==========
ae_model.eval()
logger = log.logger.RunLogger(Path(__file__).resolve().parent, config.model, config.train)
input_tensor_size = (config.train.minibatch_size, 1,
                     config.model.spectrogram_size[0], config.model.spectrogram_size[1])
logger.init_with_model(ae_model, input_tensor_size)
logger.tensorboard.init_hparams_and_metrics(metrics)


# ========== Model training epochs ==========
ae_model.train()
# TODO consider start epoch
for epoch in range(0, config.train.n_epochs):
    # TODO train all mini-batches

    # TODO evaluation on validation dataset

    # TODO epoch logs (epoch scalars/sounds/images + metrics update)
    logger.tensorboard.update_metrics(metrics)
    logger.on_epoch_finished(epoch, ae_model)


# ========== Logger final stats ==========
logger.on_training_finished()

