"""
Allows easy modification of all configuration parameters required to define,
train or evaluate a model.
This script is not intended to be run, it only describes parameters.
"""


import datetime

from model import loss


class _Config(object):
    pass


model = _Config()
model.name = "BasicVAE"
model.run_name = '00_tempDebug'  # different hyperparams, optimizer, etc... for a given model
model.allow_erase_run = True  # If True, a previous run with identical name will be erased before new training
# See model/encoder.py to view available architectures. Decoder architecture will be as symmetric as possible.
model.encoder_architecture = "wavenet_baseline"
# Spectrogram size cannot easily be modified - all CNN decoders should be re-written
model.spectrogram_size = (513, 433)  # Corresponding STFT: fft 1024 hop 256, audio 5.0s@22.05kHz
# Latent space dimension
model.dim_z = 8
# Directory for saving metrics, samples, models, etc... see README.md
model.logs_root_dir = "saved"  # Path from this directory


train = _Config()
train.start_datetime = datetime.datetime.now().isoformat()
train.minibatch_size = 8
train.start_epoch = 0  # 0 means a restart (previous data erased by the logger)
train.n_epochs = 10  # Total number of epochs (including previous training epochs)
train.save_period = 1  # Period (in epochs) for tensorboard logs and model saves
train.latent_loss = 'Dkl'  # Latent regularization loss: Dkl or MMD TODO mettre direct dans un classe de Loss
train.ae_reconstruction_loss = loss.MSELoss()
train.metrics = ['ReconsLoss']
# TODO scheduler, etc....


evaluate = _Config()
evaluate.epoch = -1  # Trained model to be loaded for post-training evaluation.


# ---------------------------------------------------------------------------------------

