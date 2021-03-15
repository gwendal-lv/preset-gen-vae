"""
Datasets of synth sounds. PyTorch-compatible, with a lot of added method and properties for synthesizer
parameters learning.
Concrete preset Datasets are available from this module but are implemented in their own files.
"""


from . import dexeddataset
from . import divadataset
# ====================== Concrete dataset classes ======================
DexedDataset = dexeddataset.DexedDataset
DivaDataset = divadataset.DivaDataset
# ======================================================================



def model_config_to_dataset_kwargs(model_config):
    """ Creates a dict that can be unpacked to pass to a PresetDataset class constructor.

    :param model_config: should be the config.model attribute from config.py. """
    return {'note_duration': model_config.note_duration, 'n_fft': model_config.stft_args[0],
            'fft_hop': model_config.stft_args[1], 'n_mel_bins': model_config.mel_bins,
            'spectrogram_min_dB': model_config.spectrogram_min_dB,
            'midi_notes': model_config.midi_notes, 'multichannel_stacked_spectrograms': model_config.stack_spectrograms}
