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
from utils.config import _Config  # Empty class - to ease JSON serialization of this file


model = _Config()
model.name = "ExtVAE2"
# TODO auto append k-fold to run name?
model.run_name = '50_dev_test'  # run: different hyperparams, optimizer, etc... for a given model
model.allow_erase_run = False  # If True, a previous run with identical name will be erased before new training
# See model/encoder.py to view available architectures. Decoder architecture will be as symmetric as possible.
model.encoder_architecture = 'speccnn8l1_bn'
# Possible values: 'flow_realnvp_4l180', 'mlp_3l1024', ... (configurable numbers of layers and neurons)
# Optional suffixes: _bn, _nobn, ... TODO implement opts
model.params_regression_architecture = 'flow_realnvp_4l200'
# Spectrogram size cannot easily be modified - all CNN decoders should be re-written
model.note_duration = (3.0, 1.0)
model.stft_args = (1024, 256)  # fft size and hop size
model.mel_bins = 257  # -1 disables Mel-scale spectrogram. Try: 257, 513, ...
model.mel_f_limits = (0, 11050)  # min/max Mel-spectrogram frequencies TODO implement
# Tuple of (pitch, velocity) tuples. Using only 1 midi note is fine.
# model.midi_notes = ((60, 85), )  # Reference note
model.midi_notes = ((40, 85), (50, 85), (60, 42), (60, 85), (60, 127), (70, 85))
model.stack_spectrograms = False
model.spectrogram_min_dB = -120.0
# Possible spectrogram sizes:
#   (513, 433): audio 5.0s, fft size 1024, fft hop 256
#   (257, 347): audio 4.0s, fft size 512 (or fft 1024 w/ mel_bins 257), fft hop 256
model.spectrogram_size = (257, 347)  # see data/dataset.py to retrieve this from audio/stft params
model.dim_z = 256  # Latent space dimension  *** When using a Flow regressor, this dim is automatically set ***
# Latent flow architecture, e.g. 'realnvp_4l200' (4 flows, 200 hidden features per flow)  TODO batch norm args
#    - base architectures can be realnvp, maf, ...
#    - set to None to disable latent space flow transforms
model.latent_flow_arch = 'realnvp_4l200'
# If True, loss compares v_out and v_in. If False, we will flow-invert v_in to get loss in the q_Z0 domain.
# This option has implications on the regression model itself (the flow will be used in direct or inverse order)
model.forward_controls_loss = True  # Must be true for non-invertible MLP regression

model.synth = 'dexed'
# Dexed-specific auto rename: '*' in 'al*_op*_lab*' will be replaced by the actual algorithms, operators and labels
model.synth_args_str = 'al*_op*_lab*'  # Auto-generated string (see end of script)
model.synth_params_count = -1  # Will be set automatically - see data.build.get_full_and_split_datasets
model.learnable_params_tensor_length = -1  # Will be set automatically - see data.build.get_full_and_split_datasets
# Modeling of synth controls probability distributions
# Possible values: None, 'vst_cat' or 'all<=xx' where xx is numerical params threshold cardinal
model.synth_vst_params_learned_as_categorical = 'all<=32'
# flags/values to describe the dataset to be used
model.dataset_labels = ('harmonic',)  # tuple of labels, or None to use all available labels
# Dexed: Preset Algorithms and activated Operators (List of ints, None to use all)
# Other synth: ...?
model.dataset_synth_args = ([1, 2, 7, 8, 9, 14], [1, 2, 3])
# Directory for saving metrics, samples, models, etc... see README.md
model.logs_root_dir = "saved"  # Path from this directory


train = _Config()
train.start_datetime = datetime.datetime.now().isoformat()
train.minibatch_size = 256
train.test_holdout_proportion = 0.2
train.k_folds = 5
train.current_k_fold = 1
train.start_epoch = 0  # 0 means a restart (previous data erased). If > 0: will load start_epoch-1 checkpoint
train.n_epochs = 1000  # Total number of epochs (including previous training epochs)
train.save_period = 20  # Period for model saves (large disk size). Tensorboard scalars/metric logs at all epochs.
train.plot_period = 10  # Period (in epochs) for plotting graphs into Tensorboard (quite CPU expensive)
train.latent_loss = 'Dkl'  # Latent regularization loss: Dkl or MMD for Basic VAE. Specific loss for Flow VAE
# Losses normalization allow to get losses in the same order of magnitude, but does not optimize the true ELBO.
# When un-normalized, the reconstruction loss (log-probability of a multivariate gaussian) is orders of magnitude
# bigger than other losses. Train does not work with normalize=False at the moment - use train.beta to compensate
train.normalize_losses = True  # Normalize all losses over the vector-dimension (e.g. spectrogram pixels count, D, ...)


# TODO train regression network alone when full-train has finished?
#    that requires two optimizers and two schedulers (one full-model, one for regression)
train.optimizer = 'Adam'
# Maximal learning rate (reached after warmup, then reduced on plateaus)
# LR decreased if non-normalized losses (which are expected to be 90,000 times bigger with a 257x347 spectrogram)
train.initial_learning_rate = 2e-4 if True else 2e-9  # e-9 LR with e+4 loss does not allow any train (vanishing grad?)
# Learning rate warmup (see https://arxiv.org/abs/1706.02677)
train.lr_warmup_epochs = 10
train.lr_warmup_start_factor = 0.1
train.adam_betas = (0.9, 0.999)  # default (0.9, 0.999)
train.weight_decay = 1e-4  # Dynamic weight decay?
train.fc_dropout = 0.2
# (beta<1, normalize=True) corresponds to (beta>>1, normalize=False) in the beta-VAE formulation (ICLR 2017)
train.beta = 0.2  # Regularization factor for the latent loss  # TODO use lower value to get closer the ELBO
train.beta_start_value = 0.1  # 0.5 works quite well (10 warmup epochs). Should not be zero (unstable training risk)
train.beta_warmup_epochs = 50  # Epochs of warmup increase from start_value to beta
train.beta_cycle_epochs = -1  # beta cyclic annealing (https://arxiv.org/abs/1903.10145). -1 deactivates TODO do

train.scheduler_name = 'ReduceLROnPlateau'  # TODO try CosineAnnealing
# Possible values: 'VAELoss' (total), 'ReconsLoss', 'Controls/BackpropLoss', ... All required losses will be summed
train.scheduler_loss = ('ReconsLoss/Backprop', 'Controls/BackpropLoss')
train.scheduler_lr_factor = 0.2
train.scheduler_patience = 15  # Longer patience with smaller datasets and quite unstable trains
train.scheduler_cooldown = 15
train.scheduler_threshold = 1e-4
# Training considered "dead" when dynamic LR reaches this value
train.early_stop_lr_threshold = train.initial_learning_rate * 1e-3

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
if model.synth == "dexed":
    if model.dataset_synth_args[0] is not None:  # Algos
        model.synth_args_str = model.synth_args_str.replace("al*", "al" +
                                                            '.'.join([str(alg) for alg in model.dataset_synth_args[0]]))
    if model.dataset_synth_args[1] is not None:  # Operators
        model.synth_args_str = model.synth_args_str.replace("_op*", "_op" +
                                                            ''.join([str(op) for op in model.dataset_synth_args[1]]))
    if model.dataset_labels is not None:  # Labels
        model.synth_args_str = model.synth_args_str.replace("_lab*", '_' +
                                                            '_'.join([label[0:4] for label in model.dataset_labels]))
else:
    raise NotImplementedError("Unknown synth prefix for model.synth '{}'".format(model.synth))


evaluate = _Config()
evaluate.epoch = -1  # Trained model to be loaded for post-training evaluation.


# ---------------------------------------------------------------------------------------

