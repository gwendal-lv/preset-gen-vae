
""" Allows easy modification of all configuration parameters required to define,
train or evaluate a model. """


class _Config(object):
    pass


model = _Config()
model.name = "TempDebugVAE"
# See encoder.py to view available architectures. Decoder architecture will be quite symmetric.
model.encoder_architecture = "wavenet_baseline"
# Spectrogram size cannot easily be modified - all CNN decoders should be re-written
model.spectrogram_size = (513, 433)  # Corresponding STFT: fft 1024 hop 256, audio 5.0s@22.05kHz
model.dim_z = 8


train = _Config()
train.n_epochs = 1000
# TODO scheduler, etc....
