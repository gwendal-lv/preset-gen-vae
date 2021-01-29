
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

        if self.architecture == 'wavenet_baseline'\
           or self.architecture == 'wavenet_baseline_lighter':
            self.mlp = nn.Linear(self.dim_z, 1024 * 2 * 4)  # Output size corresponds to the encoder
        elif self.architecture == 'wavenet_baseline_shallow':
            self.mlp = nn.Linear(self.dim_z, 1024 * 5 * 5)  # Output size corresponds to the encoder
        elif self.architecture == 'flow_synth':
            self.mlp = nn.Sequential(nn.Linear(self.dim_z, 1024), nn.ReLU(),
                                     nn.Linear(1024, 1024), nn.ReLU(),
                                     nn.Linear(1024, 64 * 17 * 14))
        else:
            raise NotImplementedError("Architecture '{}' not available".format(self.architecture))

    def forward(self, z_sampled):
        cnn_input = self.mlp(z_sampled)
        # Reshaping depends on the decoder
        if self.architecture == 'wavenet_baseline'\
           or self.architecture == 'wavenet_baseline_lighter':
            cnn_input = cnn_input.view(-1, 1024, 2, 4)
        elif self.architecture == 'wavenet_baseline_shallow':
            cnn_input = cnn_input.view(-1, 1024, 5, 5)
        elif self.architecture == 'flow_synth':
            cnn_input = cnn_input.view(-1, 64, 17, 14)
        else:
            return NotImplementedError()
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
            No activation and batch norm after the last up-conv.
            
            Issue: this architecture induces a huge number of ops within the 2 last layers.
            Unusable with reducing the spectrogram or getting 8 or 16 GPUs. '''
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

        elif self.architecture == 'wavenet_baseline_lighter':
            ''' Lighter decoder compared to wavenet baseline, but keeps an acceptable number
            of GOPs for last transpose-conv layers '''
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
                                        layer.TConv2D(128, 64, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec7'),
                                        layer.TConv2D(64, 32, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec8'),
                                        layer.TConv2D(32, 16, [5, 5], [2, 2], 2, output_padding=[0, 0],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec9'),
                                        nn.ConvTranspose2d(16, 1, [5, 5], [2, 2], 2)
                                        )

        elif self.architecture == 'wavenet_baseline_shallow':  # Inspired from wavenet_baseline
            self.dec_nn = nn.Sequential(layer.TConv2D(1024, 512, [1 ,1], [1 ,1], 0,
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec1'),
                                        layer.TConv2D(512, 256, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec2'),
                                        layer.TConv2D(256, 128, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec3'),
                                        layer.TConv2D(128, 64, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec4'),
                                        layer.TConv2D(64, 32, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec5'),
                                        layer.TConv2D(32, 16, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec6'),
                                        layer.TConv2D(16, 8, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                      activation=nn.LeakyReLU(0.1), name_prefix='dec7'),
                                        nn.ConvTranspose2d(8, 1, [5, 5], [2, 2], 2)
                                        )

        elif self.architecture == 'flow_synth':  # https://acids-ircam.github.io/flow_synthesizer/#models-details
            ''' This decoder is as GPU-heavy as wavenet_baseline '''
            n_lay = 64  # 128/2 for paper's comparisons consistency. Could be larger
            k7 = [7, 7]  # Kernel of size 7
            self.dec_nn = nn.Sequential(layer.TConv2D(n_lay, n_lay, k7, [2, 2], 3,
                                                      activation=nn.ELU(), name_prefix='dec1'),
                                        layer.TConv2D(n_lay, n_lay, k7, [2, 2], 3,
                                                      activation=nn.ELU(), name_prefix='dec2'),
                                        layer.TConv2D(n_lay, n_lay, k7, [2, 2], 3,
                                                      activation=nn.ELU(), name_prefix='dec3'),
                                        layer.TConv2D(n_lay, n_lay, k7, [2, 2], 3,
                                                      activation=nn.ELU(), name_prefix='dec4'),
                                        layer.TConv2D(n_lay, 1, k7, [2, 2], 2,
                                                      activation=nn.ELU(), name_prefix='dec5')
                                        )

        else:
            raise NotImplementedError("Architecture '{}' not available".format(self.architecture))

    def forward(self, x_spectrogram):
        return self.dec_nn(x_spectrogram)
