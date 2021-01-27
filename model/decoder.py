
import torch
import torch.nn as nn

from model import layer


class SpectrogramDecoder(nn.Module):
    """ Contains a spectrogram-input CNN and some MLP layers, and outputs the mu/logsigma2 values"""
    def __init__(self, architecture, dim_z):
        super().__init__()
        self.dim_z = dim_z  # Latent-vector size
        self.architecture = architecture
        self.cnn = SpectrogramCNN(self.architecture)

        if self.architecture == 'wavenet_baseline':
            self.mlp = nn.Linear(self.dim_z, 1024 * 2 * 4)  # Output size corresponds to the encoder
        else:
            raise NotImplementedError("Architecture '{}' not available".format(self.architecture))

    def forward(self, z_sampled):
        cnn_input = self.mlp(z_sampled)
        # Reshaping depends on the decoder
        if self.architecture == 'wavenet_baseline':
            cnn_input = cnn_input.view(-1, 1024, 2, 4)
        return self.cnn(cnn_input)


class SpectrogramCNN(nn.Module):
    """ A decoder CNN network for spectrogram output """

    # TODO Option to enable res skip connections
    # TODO Option to choose activation function
    def __init__(self, architecture):
        """ Defines a decoder given the specified architecture.
        Padding is chosen such that the last output is a few pixels larger than the target spectrogram
        """
        super().__init__()
        self.architecture = architecture

        if self.architecture == 'wavenet_baseline':  # https://arxiv.org/abs/1704.01279
            ''' Symmetric layer output sizes (compared to the encoder).
            No activation and batch norm after the last up-conv '''
            self.dec_nn = nn.Sequential(layer.TConv2D(1024, 512, [1 ,1], [1 ,1], 0,
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec1'),
                                        layer.TConv2D(512, 512, [4, 4], [2, 1], 2, output_padding=[1, 0],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec2'),
                                        layer.TConv2D(512, 256, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec3'),
                                        layer.TConv2D(256, 256, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec4'),
                                        layer.TConv2D(256, 256, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec5'),
                                        layer.TConv2D(256, 128, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec6'),
                                        layer.TConv2D(128, 128, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec7'),
                                        layer.TConv2D(128, 128, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec8'),
                                        layer.TConv2D(128, 128, [5, 5], [2, 2], 2, output_padding=[0, 0],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec9'),
                                        nn.ConvTranspose2d(128, 1, [5, 5], [2, 2], 2)
                                        )

        elif self.architecture == 'wavenet_baseline_reduced':  # Inspired from wavenet_baseline
            self.dec_nn = nn.Sequential()  # TODO
            raise NotImplementedError()

        elif self.architecture == 'flow_synth':  # https://acids-ircam.github.io/flow_synthesizer/#models-details
            ''' TODO '''
            n_lay = 64  # 128/2 for paper's comparisons consistency. Could be larger
            self.dec_nn = nn.Sequential()
            raise NotImplementedError()

        else:
            raise NotImplementedError("Architecture '{}' not available".format(self.architecture))

    def forward(self, x_spectrogram):
        return self.dec_nn(x_spectrogram)
