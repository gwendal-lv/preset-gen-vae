
import torch
import torch.nn as nn

from model import layer


def available_architectures():
    return ['wavenet_baseline', 'wavenet_baseline_reduced',
            'flow_synth']


class SpectrogramEncoder(nn.Module):
    """ Contains a spectrogram-input CNN and some MLP layers, and outputs the mu/logsigma values"""
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        # TODO init l'encodeur, puis aller voir


class SpectrogramCNN(nn.Module):
    """ A encoder CNN network for spectrogram input """

    # TODO Option to enable res skip connections
    # TODO Option to choose activation function
    def __init__(self, architecture):
        """ Automatically defines an autoencoder given the specified architecture
        """
        super().__init__()
        self.architecture = architecture
        if self.architecture not in available_architectures():
            raise NotImplementedError("Architecture '{}' not available".format(self.architecture))

        elif self.architecture == 'wavenet_baseline':  # https://arxiv.org/abs/1704.01279
            ''' Based on strided convolutions - no max pool (reduces the total amount of
             conv operations).
             No dilation: the receptive field in enlarged through a larger number
             of layers. 
             Layer 8 has a lower time-stride (better time resolution).
             Size of layer 9 (1024 ch) corresponds the wavenet time-encoder.
             
             Issue: when using the paper's FFT size and hop, layers 8 and 9 are quite useless. The image size
              at this depth is < kernel size (much of the 4x4 kernel is useless) '''
            self.enc_nn = nn.Sequential(layer.Conv2D(1, 128, [5,5], [2,2], 2, [1,1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc1'),
                                        layer.Conv2D(128, 128, [4,4], [2,2], 2, [1,1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc2'),
                                        layer.Conv2D(128, 128, [4,4], [2,2], 2, [1,1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc3'),
                                        layer.Conv2D(128, 256, [4,4], [2,2], 2, [1,1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc4'),
                                        layer.Conv2D(256, 256, [4,4], [2,2], 2, [1,1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc5'),
                                        layer.Conv2D(256, 256, [4,4], [2,2], 2, [1,1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc6'),
                                        layer.Conv2D(256, 512, [4,4], [2,2], 2, [1,1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc7'),
                                        layer.Conv2D(512, 512, [4,4], [2,2], 2, [1,1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc8'),
                                        layer.Conv2D(512, 512, [4,4], [2,1], 2, [1,1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc9'),
                                        layer.Conv2D(512, 1024, [1,1], [1,1], 0, [1,1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc10'),
                                        )

        elif self.architecture == 'wavenet_baseline_reduced':
            self.enc_nn = nn.Sequential(layer.Conv2D(1, 128, [5, 5], [2, 2], 2, [1, 1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc1'),
                                        layer.Conv2D(128, 128, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc2'),
                                        layer.Conv2D(128, 128, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc3'),
                                        layer.Conv2D(128, 256, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc4'),
                                        layer.Conv2D(256, 256, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc5'),
                                        layer.Conv2D(256, 256, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc6'),
                                        layer.Conv2D(256, 512, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc7'),
                                        layer.Conv2D(512, 1024, [1, 1], [1, 1], 0, [1, 1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc8'),
                                        )

        elif self.architecture == 'flow_synth':  # https://acids-ircam.github.io/flow_synthesizer/#models-details
            ''' Based on strided convolutions and dilation to quickly enlarge the receptive field.
            Paper says: "5 layers with 128 channels of strided dilated 2-D convolutions with kernel
            size 7, stride 2 and an exponential dilation factor of 2l (starting at l=0 with batch
            normalization and ELU activation."
            The padding is 3 * 2^l (not detailed in the paper).
            
            Potential issue: the dilation is extremely big for deep layers 4 and 5. Dilated kernel is applied
            mostly on zero-padded values. '''
            n_lay = 64  # 128/2 for paper's comparisons consistency. Could be larger
            self.enc_nn = nn.Sequential(layer.Conv2D(1, n_lay, [7,7], [2,2], 3, [1,1],
                                                     activation=nn.ELU(), name_prefix='enc1'),
                                        layer.Conv2D(n_lay, n_lay, [7, 7], [2, 2], 6, [2, 2],
                                                     activation=nn.ELU(), name_prefix='enc2'),
                                        layer.Conv2D(n_lay, n_lay, [7, 7], [2, 2], 12, [4, 4],
                                                     activation=nn.ELU(), name_prefix='enc3'),
                                        layer.Conv2D(n_lay, n_lay, [7, 7], [2, 2], 24, [8, 8],
                                                     activation=nn.ELU(), name_prefix='enc4'),
                                        layer.Conv2D(n_lay, n_lay, [7, 7], [2, 2], 48, [16, 16],
                                                     activation=nn.ELU(), name_prefix='enc5'))

    def forward(self, x_spectrogram):
        return self.enc_nn(x_spectrogram)


if __name__ == "__main__":

    # Test: does the dataloader get stuck here as well? Or only in jupyter notebooks?
    # YEP lots of multiprocessing issues with PyTorch DataLoaders....
    # (even pickle could be an issue... )
    from data import dataset
    import torchinfo
    dexed_dataset = dataset.DexedDataset()

    enc = SpectrogramCNN(architecture='wavenet_baseline')
    dataloader = torch.utils.data.DataLoader(dexed_dataset, batch_size=32, shuffle=False, num_workers=40)
    spectro, params, midi = next(iter(dataloader))
    print("Input spectrogram tensor: {}".format(spectro.size()))
    encoded_spectrogram = enc(spectro)
    _ = torchinfo.summary(enc, input_size=spectro.size())


