"""
Allows easy modification of all configuration parameters required to define,
train or evaluate a model.
This script is not intended to be run, it only describes parameters.
"""


import datetime


class _Config(object):
    pass


model = _Config()
model.name = "BasicVAE"
model.run_name = '03_testsave'  # run: different hyperparams, optimizer, etc... for a given model
model.allow_erase_run = False  # If True, a previous run with identical name will be erased before new training
# See model/encoder.py to view available architectures. Decoder architecture will be as symmetric as possible.
model.encoder_architecture = 'speccnn8l1'
# Spectrogram size cannot easily be modified - all CNN decoders should be re-written
model.note_duration = (3.0, 1.0)
model.stft_args = (512, 256)  # fft size and hop size
# Possible spectrogram sizes:
# (513, 433): audio 5.0s, fft size 1024, fft hop 256
# (257, 347): audio 4.0s, fft size 512, fft hop 256
model.spectrogram_size = (257, 347)  # see data/dataset.py to retrieve this from audio/stft params
# Latent space dimension
model.dim_z = 256 #trololo
# Directory for saving metrics, samples, models, etc... see README.md
model.logs_root_dir = "saved"  # Path from this directory


train = _Config()
train.start_datetime = datetime.datetime.now().isoformat()
train.minibatch_size = 256
train.datasets_proportions = [0.8, 0.1, 0.1]  # train/validation/test sub-datasets sizes (total must be 1.0)
train.start_epoch = 10  # 0 means a restart (previous data erased). >0: will load start_epoch-1 saved model
train.n_epochs = 13  # Total number of epochs (including previous training epochs)
train.save_period = 1  # Period (in epochs) for tensorboard logs and model saves
train.latent_loss = 'Dkl'  # Latent regularization loss: Dkl or MMD
train.normalize_latent_loss = True  # Normalize the latent over z-dimension
train.ae_reconstruction_loss = 'MSE'
train.metrics = ['ReconsLoss', 'LatLoss']

train.optimizer = 'Adam'
train.initial_learning_rate = 3e-4
train.weight_decay = 1e-5  # Dynamic weight decay?
train.beta = 1.0  # Regularization factor for the latent loss
train.beta_start_value = 0.2
train.beta_warmup_epochs = 20  # Epochs of warmup increase from 0.0 to beta

train.scheduler_name = 'ReduceLROnPlateau'
train.scheduler_lr_factor = 0.2
train.scheduler_patience = 5
train.scheduler_threshold = 1e-3

train.verbosity = 2  # 0: no console output --> 3: fully-detailed per-batch console output
train.init_security_pause = 0.0  # Short pause before erasing an existing run
train.logged_samples_count = 4  # Number of logged audio and spectrograms for a given epoch
train.logged_samples_period = 10  # Epoch periods (not to store too much .wav/.png data)
train.profiler_args = {'enabled': False, 'use_cuda': True, 'record_shapes': False,
                       'profile_memory': False, 'with_stack': False}
train.profiler_full_trace = False  # If True, runs only a few batches then exits - but saves a fully detailed trace.json
train.profiler_1_GPU = False  # Profiling on only 1 GPU allow a much better understanding of trace.json


# Mini-batch size can be smaller for the last mini-batches and/or during evaluation
model.input_tensor_size = (train.minibatch_size, 1, model.spectrogram_size[0], model.spectrogram_size[1])


evaluate = _Config()
evaluate.epoch = -1  # Trained model to be loaded for post-training evaluation.


# ---------------------------------------------------------------------------------------

