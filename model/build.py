"""
Utility functions for building a new model (using only config from config.py),
or for building a previously trained model.
"""

from model import VAE, encoder, decoder, extendedAE, regression


def build_ae_model(model_config, train_config):
    """
    Builds an auto-encoder model given a configuration. Built model can be initialized later
    with a previous state_dict.

    :param model_config: model global attribute from the config.py module
    :param train_config: train attributes (a few are required, e.g. dropout probability)
    :return: Tuple: encoder, decoder, full AE model
    """
    # Encoder and decoder with the same architecture
    encoder_model = encoder.SpectrogramEncoder(model_config.encoder_architecture, model_config.dim_z,
                                               model_config.spectrogram_size, train_config.fc_dropout)
    decoder_model = decoder.SpectrogramDecoder(model_config.encoder_architecture, model_config.dim_z,
                                               model_config.spectrogram_size, train_config.fc_dropout)
    # AE model, not parallelized (if required, this should be done afterwards)
    if model_config.latent_flow_arch is None:
        ae_model = VAE.BasicVAE(encoder_model, model_config.dim_z, decoder_model, train_config.normalize_losses,
                                train_config.latent_loss)
    else:
        # TODO flow dropout (in all but the last flow layers)
        ae_model = VAE.FlowVAE(encoder_model, model_config.dim_z, decoder_model, train_config.normalize_losses,
                               model_config.latent_flow_arch)
    return encoder_model, decoder_model, ae_model


def build_extended_ae_model(model_config, train_config, idx_helper):
    """ Builds a spectral auto-encoder model, and a synth parameters regression model which takes
    latent vectors as input. Both models are integrated into an ExtendedAE model. """
    # Spectral VAE
    encoder_model, decoder_model, ae_model = build_ae_model(model_config, train_config)
    # Regression model
    if model_config.params_regression_architecture.startswith("mlp_"):
        assert model_config.forward_controls_loss is True  # Non-invertible MLP cannot inverse target values
        reg_arch = model_config.params_regression_architecture.replace("mlp_", "")
        reg_model = regression.MLPRegression(reg_arch, model_config.dim_z, idx_helper, train_config.fc_dropout)
    elif model_config.params_regression_architecture.startswith("flow_"):
        assert model_config.learnable_params_tensor_length > 0  # Flow models require dim_z to be equal to this length
        reg_arch = model_config.params_regression_architecture.replace("flow_", "")
        reg_model = regression.FlowRegression(reg_arch, model_config.dim_z, idx_helper,
                                              fast_forward_flow=model_config.forward_controls_loss,
                                              dropout_p=train_config.fc_dropout)
    else:
        raise NotImplementedError("Synth param regression arch '{}' not implemented"
                                  .format(model_config.params_regression_architecture))
    extended_ae_model = extendedAE.ExtendedAE(ae_model, reg_model, idx_helper, train_config.fc_dropout)
    return encoder_model, decoder_model, ae_model, extended_ae_model


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
    # Model config check TODO add new attributes to check
    prev_config = config_json_checkpoint['model']
    attributes_to_check = ['name', 'run_name', 'encoder_architecture', 'dim_z', 'logs_root_dir',
                           'note_duration', 'stft_args', 'spectrogram_size', 'mel_bins']
    for attr in attributes_to_check:
        if not _is_attr_equal(prev_config[attr], new_model_config.__dict__[attr]):
            raise ValueError("Model attribute '{}' is different in the new config.py ({}) and the old config.json ({})"
                             .format(attr, new_model_config.__dict__[attr], prev_config[attr]))
    # Train config check TODO add new attributes to check
    prev_config = config_json_checkpoint['train']
    attributes_to_check = ['minibatch_size', 'datasets_proportions', 'latent_loss', 'normalize_latent_loss',
                           'ae_reconstruction_loss', 'optimizer', 'scheduler_name']
    for attr in attributes_to_check:
        if not _is_attr_equal(prev_config[attr], new_train_config.__dict__[attr]):
            raise ValueError("Train attribute '{}' is different in the new config.py ({}) and the old config.json ({})"
                             .format(attr, new_train_config.__dict__[attr], prev_config[attr]))


# Model Build tests - see also params_regression.ipynb
if __name__ == "__main__":

    import sys
    import pathlib  # Dirty path trick to import config.py from project root dir
    sys.path.append(pathlib.Path(__file__).parent.parent)
    import config

    # Manual config changes (for test purposes only)
    config.model.synth_params_count = 144

    _encoder_model, _decoder_model, _ae_model, _extended_ae_model \
        = build_extended_ae_model(config.model, config.train)
