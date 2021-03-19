
import numpy as np
import torch
import torch.nn as nn

from model import layer


class SpectrogramDecoder(nn.Module):
    """ Contains a spectrogram-input CNN and some MLP layers, and outputs the mu/logsigma2 values"""
    def __init__(self, architecture, dim_z, output_tensor_size, fc_dropout):
        super().__init__()
        # TODO test bigger output spectrogram - with final centered crop
        self.output_tensor_size = output_tensor_size
        # Encoder input size is desired output size for this decoder (crop if too big? not necessary at the moment)
        self.spectrogram_input_size = (self.output_tensor_size[2], self.output_tensor_size[3])
        self.spectrogram_channels = output_tensor_size[1]
        self.dim_z = dim_z  # Latent-vector size
        self.architecture = architecture
        self.cnn_input_shape = None  # shape not including batch size
        self.mixer_1x1conv_ch = 2048
        self.fc_dropout = fc_dropout

        if 'speccnn8l1' not in self.architecture:
            raise NotImplementedError("Only speccnn8l1 is currently available"
                                      "(stacked multi-note spectograms compatibility)")

        # - - - - - 1) MLP output size must to correspond to encoder's MLP input size - - - -
        if self.architecture == 'wavenet_baseline'\
           or self.architecture == 'wavenet_baseline_lighter':
            assert self.spectrogram_input_size == (513, 433)  # Big spectrogram only - TODO adapt
            self.cnn_input_shape = (1024, 2, 4)
            self.mlp = nn.Linear(self.dim_z, int(np.prod(self.cnn_input_shape)))
        elif self.architecture == 'wavenet_baseline_shallow':
            assert self.spectrogram_input_size == (513, 433)  # Big spectrogram only - TODO adapt
            self.cnn_input_shape = (1024, 5, 5)
            self.mlp = nn.Linear(self.dim_z, int(np.prod(self.cnn_input_shape)))
        elif self.architecture == 'flow_synth':
            if self.spectrogram_input_size == (513, 433):
                self.cnn_input_shape = (64, 17, 14)
            elif self.spectrogram_input_size == (257, 347):
                self.cnn_input_shape = (64, 3, 6)
            self.mlp = nn.Sequential(nn.Linear(self.dim_z, 1024), nn.ReLU(),  # TODO dropout
                                     nn.Linear(1024, 1024), nn.ReLU(),
                                     nn.Linear(1024, int(np.prod(self.cnn_input_shape))))  # TODO add last ReLU?
        elif 'speccnn8l1' in self.architecture:
            if self.spectrogram_input_size == (257, 347):
                if self.architecture == 'speccnn8l1_3':
                    self.cnn_input_shape = (self.mixer_1x1conv_ch, 3, 3)
                else:
                    self.cnn_input_shape = (self.mixer_1x1conv_ch, 3, 4)
                # No ReLU (encoder-symmetry) (and leads to very bad generalization, but don't know why)
                self.mlp = nn.Sequential(nn.Linear(self.dim_z, int(np.prod(self.cnn_input_shape))),
                                         nn.Dropout(self.fc_dropout))
            else:
                assert NotImplementedError()
        else:
            raise NotImplementedError("Architecture '{}' not available".format(self.architecture))

        # - - - - - 2) Features "un-mixer" - - - - -
        self.features_unmixer_cnn = layer.TConv2D(self.mixer_1x1conv_ch, self.spectrogram_channels*512,
                                                  [1, 1], [1, 1], 0,
                                                  activation=nn.LeakyReLU(0.1), name_prefix='dec1')

        # - - - - - 3) Main CNN decoder (applied once per spectrogram channel) - - - - -
        single_spec_size = list(self.spectrogram_input_size)
        single_spec_size[1] = 1
        self.single_ch_cnn = SpectrogramCNN(self.architecture, single_spec_size, append_1x1_conv=False)

    def forward(self, z_sampled):
        mixed_features = self.mlp(z_sampled)
        mixed_features = mixed_features.view(-1,  # batch size auto inferred
                                             self.cnn_input_shape[0], self.cnn_input_shape[1], self.cnn_input_shape[2])
        unmixed_features = self.features_unmixer_cnn(mixed_features)
        single_ch_cnn_inputs = torch.split(unmixed_features, 512, dim=1)  # Split along channels dimension
        single_ch_cnn_outputs = [self.single_ch_cnn(single_ch_in) for single_ch_in in single_ch_cnn_inputs]
        return torch.cat(single_ch_cnn_outputs, dim=1)  # Concatenate all single-channel spectrograms


class SpectrogramCNN(nn.Module):
    """ A decoder CNN network for spectrogram output """

    def __init__(self, architecture, spectrogram_input_size, output_activation=nn.Hardtanh(), append_1x1_conv=True):
        """ Defines a decoder given the specified architecture. """
        super().__init__()
        self.architecture = architecture
        if not append_1x1_conv:
            assert self.architecture == 'speccnn8l1_bn'  # Only this arch is fully-supported at the moment
        self.spectrogram_input_size = spectrogram_input_size
        assert self.spectrogram_input_size[1] == 1  # This decoder is single-channel output
        output_activation

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
                                        output_activation
                                        )

        elif self.architecture == 'speccnn8l1'\
            or self.architecture == 'speccnn8l1_bn':  # 1.8 GB (RAM) ; 0.36 GMultAdd  (batch 256)
            ''' Inspired by the wavenet baseline spectral autoencoder, but all sizes drastically reduced '''
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            self.dec_nn = nn.Sequential(layer.TConv2D(512, 256, [4, 4], [2, 2], 2, output_padding=[1, 1],
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
                                        output_activation
                                        )
            if append_1x1_conv:  # 1x1 "un-mixing" conv inserted as first conv layer
                assert False  # FIXME 1024ch should not be constant
                self.dec_nn = nn.Sequential(layer.TConv2D(1024, 512, [1 ,1], [1 ,1], 0,
                                                          activation=act(act_p), name_prefix='dec1'),
                                            self.dec_nn)

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
                                        output_activation
                                        )
        elif self.architecture == 'speccnn8l1_3':  # XXX GB (RAM) ; XXX GMultAdd  (batch 256)
            ''' Inspired by the wavenet baseline spectral autoencoder, but all sizes drastically reduced '''
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            ker = [5, 5]
            self.dec_nn = nn.Sequential(layer.TConv2D(1024, 512, [1, 1], [1, 1], 0,
                                                      activation=act(act_p), name_prefix='dec1'),
                                        layer.TConv2D(512, 256, ker, [2, 2], 2, output_padding=[0, 1],
                                                      activation=act(act_p), name_prefix='dec2'),
                                        layer.TConv2D(256, 128, ker, [2, 2], 2, output_padding=[0, 0],
                                                      activation=act(act_p), name_prefix='dec3'),
                                        layer.TConv2D(128, 64, ker, [2, 2], 2, output_padding=[0, 1],
                                                      activation=act(act_p), name_prefix='dec4'),
                                        layer.TConv2D(64, 32, ker, [2, 2], 2, output_padding=[0, 1],
                                                      activation=act(act_p), name_prefix='dec5'),
                                        layer.TConv2D(32, 16, ker, [2, 2], 2, output_padding=[0, 0],
                                                      activation=act(act_p), name_prefix='dec6'),
                                        layer.TConv2D(16, 8, ker, [2, 2], 2, output_padding=[0, 1],
                                                      activation=act(act_p), name_prefix='dec7'),
                                        nn.ConvTranspose2d(8, 1, [5, 5], [2, 2], 2),
                                        output_activation
                                        )

        else:
            raise NotImplementedError("Architecture '{}' not available".format(self.architecture))

    def forward(self, x_spectrogram):
        return self.dec_nn(x_spectrogram)
