"""
Performs training for the configuration described in config.py
"""

import torchinfo

import config
from model import VAE, encoder, decoder


# ========== Model definition ==========
# Encoder and decoder with the same architecture
encoder_model = encoder.SpectrogramEncoder(config.model.encoder_architecture, config.model.dim_z,
                                           config.model.spectrogram_size)
decoder_model = decoder.SpectrogramDecoder(config.model.encoder_architecture, config.model.dim_z)
ae_model = VAE.BasicVAE(encoder_model, config.model.dim_z, decoder_model)

