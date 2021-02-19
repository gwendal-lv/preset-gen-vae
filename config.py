"""
Allows easy modification of all configuration parameters required to define,
train or evaluate a model.
This script is not intended to be run, it only describes parameters.

This configuration is used when running train.py as main.
When running train_queue.py, configuration changes are relative to this config.py file.

When a run starts, this file is stored as a config.json file. To ensure easy restoration of
parameters, please only use simple types such as string, ints, floats, tuples (no lists) and dicts.
"""


import datetime
from utils.config import _Config  # Empty class


model = _Config()
model.name = "ExtVAE0"
model.run_name = '07_dev_test'  # run: different hyperparams, optimizer, etc... for a given model
model.allow_erase_run = False  # If True, a previous run with identical name will be erased before new training
# See model/encoder.py to view available architectures. Decoder architecture will be as symmetric as possible.
model.encoder_architecture = 'speccnn8l1_bn'
model.params_regression = 'mlp'  # Parameters regression model
model.params_regression_architecture = '3l1024'
# Spectrogram size cannot easily be modified - all CNN decoders should be re-written
model.note_duration = (3.0, 1.0)
model.stft_args = (1024, 256)  # fft size and hop size
model.mel_bins = 257  # -1 disables Mel-scale spectrogram. Try: 257, 513, ...
model.mel_f_limits = (0, 11050)  # min/max Mel-spectrogram frequencies TODO implement
model.spectrogram_min_dB = -120.0
# Possible spectrogram sizes:
# (513, 433): audio 5.0s, fft size 1024, fft hop 256
# (257, 347): audio 4.0s, fft size 512 (or fft 1024 w/ mel_bins 257), fft hop 256
model.spectrogram_size = (257, 347)  # see data/dataset.py to retrieve this from audio/stft params
# Latent space dimension
model.dim_z = 256
# Modeling of synth controls probability distributions
model.controls_losses = 'MSE'  # MSE-only, or MSE for continuous controls and Categorical for discrete
# Synth used. Dexed-specific auto rename: '*' will be replaced by the actual algorithms, operators and labels
model.synth = 'dexed_al*_op*_lab*'
model.synth_params_count = -1  # Will be inferred automatically from a constructed dataset TODO implement
# flags/values to describe the dataset to be used
model.dataset_labels = ('harmonic',)  # tuple of labels, or None to use all available labels
# Dexed: Preset Algorithms and activated Operators (List of ints, None to use all)
# Other synth: ...?
model.dataset_synth_args = ([2], [1, 2, 3, 4, 5])
# Directory for saving metrics, samples, models, etc... see README.md
model.logs_root_dir = "saved"  # Path from this directory


train = _Config()
train.start_datetime = datetime.datetime.now().isoformat()
train.minibatch_size = 256
train.datasets_proportions = (0.8, 0.1, 0.1)  # train/validation/test sub-datasets sizes (total must be 1.0)
train.k_folds = 5  # TODO implement
train.current_k_fold = 1  # TODO implement
train.start_epoch = 0  # 0 means a restart (previous data erased). If > 0: will load start_epoch-1 checkpoint
train.n_epochs = 200  # Total number of epochs (including previous training epochs)
train.save_period = 20  # Period for model saves (large disk size). Tensorboard scalars/metric logs at all epochs.
train.plot_period = 10  # Period (in epochs) for plotting graphs into Tensorboard (quite CPU expensive)
train.latent_loss = 'Dkl'  # Latent regularization loss: Dkl or MMD
train.normalize_latent_loss = True  # Normalize the latent over z-dimension
train.ae_reconstruction_loss = 'MSE'  # TODO try spectral convergence?
# TODO loss types for controls... different losses for learning and evaluation
train.metrics = ('ReconsLoss', 'LatLoss')  # unused... metrics currently hardcoded in train.py

train.optimizer = 'Adam'
train.initial_learning_rate = 2e-4
train.adam_betas = (0.9, 0.999)  # default (0.9, 0.999)
train.weight_decay = 1e-4  # Dynamic weight decay?
train.fc_dropout = 0.2
train.beta = 1.0  # Regularization factor for the latent loss
train.beta_start_value = 0.5
train.beta_warmup_epochs = 10  # Epochs of warmup increase from 0.0 to beta
train.beta_cycle_epochs = -1  # beta cyclic annealing (https://arxiv.org/abs/1903.10145). -1 deactivates TODO do

train.scheduler_name = 'ReduceLROnPlateau'  # TODO CosineAnnealing
# Possible values: 'VAELoss' (total), 'ReconsLoss', 'ContLoss'... All required losses will be summed
train.scheduler_loss = ('ReconsLoss', 'ContLoss')
train.scheduler_lr_factor = 0.2
train.scheduler_patience = 15  # Longer patience with smaller datasets
train.scheduler_threshold = 1e-3
train.early_stop_lr_threshold = 1e-7  # Training considered "dead" when dynamic LR reaches this value

train.verbosity = 2  # 0: no console output --> 3: fully-detailed per-batch console output
train.init_security_pause = 0.0  # Short pause before erasing an existing run
train.logged_samples_count = 4  # Number of logged audio and spectrograms for a given epoch
train.logged_samples_period = 10  # Epoch periods (not to store too much .wav/.png data)
train.profiler_args = {'enabled': False, 'use_cuda': True, 'record_shapes': False,
                       'profile_memory': False, 'with_stack': False}
train.profiler_full_trace = False  # If True, runs only a few batches then exits - but saves a fully detailed trace.json
train.profiler_1_GPU = False  # Profiling on only 1 GPU allow a much better understanding of trace.json


# Mini-batch size can be smaller for the last mini-batches and/or during evaluation
# TODO multi-channel spectrograms
model.input_tensor_size = (train.minibatch_size, 1, model.spectrogram_size[0], model.spectrogram_size[1])


# Automatic model.synth string update - to retrieve this info in 1 Tensorboard string hparam
if model.synth.startswith("dexed"):
    if model.dataset_synth_args[0] is not None:  # Algos
        model.synth = model.synth.replace("_al*", "_al" + '-'.join([str(alg) for alg in model.dataset_synth_args[0]]))
    if model.dataset_synth_args[1] is not None:  # Operators
        model.synth = model.synth.replace("_op*", "_op" + ''.join([str(op) for op in model.dataset_synth_args[1]]))
    if model.dataset_labels is not None:  # Labels
        model.synth = model.synth.replace("_lab*", '_' + '_'.join([label[0:4] for label in model.dataset_labels]))
else:
    raise NotImplementedError("Unknown synth prefix for model.synth '{}'".format(model.synth))


evaluate = _Config()
evaluate.epoch = -1  # Trained model to be loaded for post-training evaluation.


# ---------------------------------------------------------------------------------------

