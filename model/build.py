"""
Utility functions for building a new model (using only config from config.py),
or for building a previously trained model.
"""

from model import VAE, encoder, decoder


def build_ae_model(model_config):
    """
    Builds an auto-encoder model given a configuration. Built model can be initialized later
    with a previous state_dict.

    :param model_config: model global attribute from the config.py module
    :return: Tuple: encoder, decoder, full AE model
    """
    # Encoder and decoder with the same architecture
    encoder_model = encoder.SpectrogramEncoder(model_config.encoder_architecture, model_config.dim_z,
                                               model_config.spectrogram_size)
    decoder_model = decoder.SpectrogramDecoder(model_config.encoder_architecture, model_config.dim_z,
                                               model_config.spectrogram_size)
    ae_model = VAE.BasicVAE(encoder_model, model_config.dim_z, decoder_model)  # Not parallelized yet
    return encoder_model, decoder_model, ae_model


def _is_attr_equal(attr1, attr2):
    """ Compares two config attributes - lists auto converted to tuples. """
    _attr1 = tuple(attr1) if isinstance(attr1, list) else attr1
    _attr2 = tuple(attr2) if isinstance(attr2, list) else attr2
    return _attr1 == _attr2


def check_configs_on_resume_from_checkpoint(new_model_config, new_train_config, config_json_checkpoint):
    """
    Performs a full consistency check between the last checkpoint saved config (stored into a .json file)
    and the new required config as described in config.py

    :raises: ValueError if any incompatibility is found

    :param new_model_config: model Class of the config.py file
    :param new_train_config: train Class of the config.py file
    :param config_json_checkpoint: config.py attributes from previous run, loaded from the .json file
    :return:
    """
    # Model config check
    prev_config = config_json_checkpoint['model']
    attributes_to_check = ['name', 'run_name', 'encoder_architecture', 'dim_z', 'logs_root_dir',
                           'note_duration', 'stft_args', 'spectrogram_size', 'mel_bins']
    for attr in attributes_to_check:
        if not _is_attr_equal(prev_config[attr], new_model_config.__dict__[attr]):
            raise ValueError("Model attribute '{}' is different in the new config.py ({}) and the old config.json ({})"
                             .format(attr, new_model_config.__dict__[attr], prev_config[attr]))
    # Train config check
    prev_config = config_json_checkpoint['train']
    attributes_to_check = ['minibatch_size', 'datasets_proportions', 'latent_loss', 'normalize_latent_loss',
                           'ae_reconstruction_loss', 'optimizer', 'scheduler_name']
    for attr in attributes_to_check:
        if not _is_attr_equal(prev_config[attr], new_train_config.__dict__[attr]):
            raise ValueError("Train attribute '{}' is different in the new config.py ({}) and the old config.json ({})"
                             .format(attr, new_train_config.__dict__[attr], prev_config[attr]))

