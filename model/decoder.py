
import numpy as np
import torch
import torch.nn as nn

from model import layer


class SpectrogramDecoder(nn.Module):
    """ Contains a spectrogram-input CNN and some MLP layers, and outputs the mu/logsigma2 values"""
    def __init__(self, architecture, dim_z, spectrogram_input_size):
        super().__init__()
        self.spectrogram_input_size = spectrogram_input_size
        self.dim_z = dim_z  # Latent-vector size
        self.architecture = architecture
        self.cnn = SpectrogramCNN(self.architecture, self.spectrogram_input_size)
        self.cnn_input_shape = None  # shape not including batch size

        # MLP output size must to correspond to encoder MLP's input size
        if self.architecture == 'wavenet_baseline'\
           or self.architecture == 'wavenet_baseline_lighter':
            assert spectrogram_input_size == (513, 433)  # Big spectrogram only - TODO adapt
            self.cnn_input_shape = (1024, 2, 4)
            self.mlp = nn.Linear(self.dim_z, int(np.prod(self.cnn_input_shape)))
        elif self.architecture == 'wavenet_baseline_shallow':
            assert spectrogram_input_size == (513, 433)  # Big spectrogram only - TODO adapt
            self.cnn_input_shape = (1024, 5, 5)
            self.mlp = nn.Linear(self.dim_z, int(np.prod(self.cnn_input_shape)))
        elif self.architecture == 'flow_synth':
            if spectrogram_input_size == (513, 433):
                self.cnn_input_shape = (64, 17, 14)
            elif spectrogram_input_size == (257, 347):
                self.cnn_input_shape = (64, 3, 6)
            self.mlp = nn.Sequential(nn.Linear(self.dim_z, 1024), nn.ReLU(),
                                     nn.Linear(1024, 1024), nn.ReLU(),
                                     nn.Linear(1024, int(np.prod(self.cnn_input_shape))))
        elif 'speccnn8l1' in self.architecture:
            if spectrogram_input_size == (257, 347):
                self.cnn_input_shape = (1024, 3, 4)
                self.mlp = nn.Linear(self.dim_z, int(np.prod(self.cnn_input_shape)))
            else:
                assert NotImplementedError()
        else:
            raise NotImplementedError("Architecture '{}' not available".format(self.architecture))

    def forward(self, z_sampled):
        cnn_input = self.mlp(z_sampled)
        cnn_input = cnn_input.view(-1, self.cnn_input_shape[0], self.cnn_input_shape[1], self.cnn_input_shape[2])
        # TODO test bigger output spectrogram - with final centered crop
        return self.cnn(cnn_input)


class SpectrogramCNN(nn.Module):
    """ A decoder CNN network for spectrogram output """

    def __init__(self, architecture, spectrogram_input_size):
        """ Defines a decoder given the specified architecture. """
        super().__init__()
        self.architecture = architecture
        self.spectrogram_input_size = spectrogram_input_size

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
                                        nn.ConvTranspose2d(128, 1, [5, 5], [2, 2], 2)  # TODO bounded activation
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
                                        nn.ConvTranspose2d(16, 1, [5, 5], [2, 2], 2)  # TODO bounded activation
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
                                        nn.ConvTranspose2d(8, 1, [5, 5], [2, 2], 2)  # TODO bounded activation
                                        )

        elif self.architecture == 'flow_synth':
            ''' This decoder seems as GPU-heavy as wavenet_baseline?? '''
            n_lay = 64  # 128/2 for paper's comparisons consistency. Could be larger
            k7 = [7, 7]  # Kernel of size 7
            if spectrogram_input_size == (513, 433):
                pads = [3, 3, 3, 3, 2]  # FIXME
                out_pads = None
            elif spectrogram_input_size == (257, 347):  # 7.7 GB (RAM), 6.0 GMultAdd (batch 256) (inc. linear layers)
                pads = [3, 3, 3, 3, 2]
                out_pads = [0, [1, 0], [0, 1], [1, 0]]  # No output padding on last layer
            self.dec_nn = nn.Sequential(layer.TConv2D(n_lay, n_lay, k7, [2, 2], pads[0], out_pads[0], [2, 2],
                                                      activation=nn.ELU(), name_prefix='dec1'),
                                        layer.TConv2D(n_lay, n_lay, k7, [2, 2], pads[1], out_pads[1], [2, 2],
                                                      activation=nn.ELU(), name_prefix='dec2'),
                                        layer.TConv2D(n_lay, n_lay, k7, [2, 2], pads[2], out_pads[2], [2, 2],
                                                      activation=nn.ELU(), name_prefix='dec3'),
                                        layer.TConv2D(n_lay, n_lay, k7, [2, 2], pads[3], out_pads[3], [2, 2],
                                                      activation=nn.ELU(), name_prefix='dec4'),
                                        nn.ConvTranspose2d(n_lay, 1, k7, [2, 2], pads[4]),
                                        nn.Tanh()
                                        )

        elif self.architecture == 'speccnn8l1':  # 1.8 GB (RAM) ; 0.36 GMultAdd  (batch 256)
            ''' Inspired by the wavenet baseline spectral autoencoder, but all sizes drastically reduced '''
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            self.dec_nn = nn.Sequential(layer.TConv2D(1024, 512, [1 ,1], [1 ,1], 0,
                                                      activation=act(act_p), name_prefix='dec1'),
                                        layer.TConv2D(512, 256, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                      activation=act(act_p), name_prefix='dec2'),
                                        layer.TConv2D(256, 128, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                      activation=act(act_p), name_prefix='dec3'),
                                        layer.TConv2D(128, 64, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                      activation=act(act_p), name_prefix='dec4'),
                                        layer.TConv2D(64, 32, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                      activation=act(act_p), name_prefix='dec5'),
                                        layer.TConv2D(32, 16, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                      activation=act(act_p), name_prefix='dec6'),
                                        layer.TConv2D(16, 8, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                      activation=act(act_p), name_prefix='dec7'),
                                        nn.ConvTranspose2d(8, 1, [5, 5], [2, 2], 2),
                                        nn.Tanh()
                                        )

        elif self.architecture == 'speccnn8l1_2':  # 5.8 GB (RAM) ; 2.4 GMultAdd  (batch 256)
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            self.dec_nn = nn.Sequential(layer.TConv2D(1024, 512, [1, 1], [1, 1], 0,
                                                      activation=act(act_p), name_prefix='dec1'),
                                        layer.TConv2D(512, 256, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                      activation=act(act_p), name_prefix='dec2'),
                                        layer.TConv2D(256, 256, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                      activation=act(act_p), name_prefix='dec3'),
                                        layer.TConv2D(256, 128, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                      activation=act(act_p), name_prefix='dec4'),
                                        layer.TConv2D(128, 128, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                      activation=act(act_p), name_prefix='dec5'),
                                        layer.TConv2D(128, 64, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                      activation=act(act_p), name_prefix='dec6'),
                                        layer.TConv2D(64, 32, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                      activation=act(act_p), name_prefix='dec7'),
                                        nn.ConvTranspose2d(32, 1, [5, 5], [2, 2], 2),
                                        nn.Tanh()
                                        )

        else:
            raise NotImplementedError("Architecture '{}' not available".format(self.architecture))

    def forward(self, x_spectrogram):
        return self.dec_nn(x_spectrogram)
