
import torch
import torch.nn as nn

from model import layer


def available_architectures():
    # TODO try a resnet CNN
    return ['wavenet_baseline',  # Ultra-heavy decoder (unusable)
            # Lighter decoder (fewer feature maps on last dec layers)
            # Remains quite heavy: >1.3 seconds to process a minibatch of 32 samples
            'wavenet_baseline_lighter',
            'wavenet_baseline_shallow',  # 8 layers instead of 10 - brutally reduced feature maps count
            'flow_synth',
            'speccnn8l1',  # Custom 8-layer CNN + 1 linear very light architecture
            'speccnn8l1_bn'  # Same base config, different BN usage (no BN on first/last layers)
            'speccnn8l1_2',  # MUCH more channels per layer.... but no significant perf improvement
            'speccnn8l1_3'  # speccnn8l1_bn with Bigger conv kernels
            ]


class SpectrogramEncoder(nn.Module):
    """ Contains a spectrogram-input CNN and some MLP layers, and outputs the mu and logs(var) values"""
    def __init__(self, architecture, dim_z, input_tensor_size, fc_dropout, output_bn=False,
                 deepest_features_mix=True, force_bigger_network=False):
        """

        :param architecture:
        :param dim_z:
        :param input_tensor_size:
        :param fc_dropout:
        :param output_bn:
        :param deepest_features_mix: (applies to multi-channel spectrograms only) If True, features mixing will be
            done on the 1x1 deepest conv layer. If False, mixing will be done before the deepest conv layer (see
            details in implementation)
        :param force_bigger_network: Optional, to impose a higher number of channels for the last 4x4 (should be
            used for fair comparisons between single/multi-specs encoder)
        """
        super().__init__()
        self.dim_z = dim_z  # Latent-vector size (2*dim_z encoded values - mu and logs sigma 2)
        self.spectrogram_channels = input_tensor_size[1]
        self.architecture = architecture
        self.deepest_features_mix = deepest_features_mix
        # 2048 if single-ch, 1024 if multi-channel 4x4 mixer (to compensate for the large number of added params)
        self.mixer_1x1conv_ch = 1024 if (self.spectrogram_channels > 1) else 2048
        self.fc_dropout = fc_dropout
        # - - - - - 1) Main CNN encoder (applied once per input spectrogram channel) - - - - -
        # stacked spectrograms: don't add the final 1x1 conv layer, or the 2 last conv layers (1x1 and 4x4)
        self.single_ch_cnn = SpectrogramCNN(self.architecture, last_layers_to_remove=(1 if self.deepest_features_mix
                                                                                      else 2))
        # - - - - - 2) Features mixer - - - - -
        assert self.architecture == 'speccnn8l1_bn'  # Only this arch is fully-supported at the moment
        self.features_mixer_cnn = nn.Sequential()
        if self.deepest_features_mix:
            self.features_mixer_cnn = layer.Conv2D(512*self.spectrogram_channels, self.mixer_1x1conv_ch,
                                                   [1, 1], [1, 1], 0, [1, 1],
                                                   activation=nn.LeakyReLU(0.1), name_prefix='enc8', batch_norm=None)
        else:  # mixing conv layer: deepest-1 (4x4 kernel)
            if not force_bigger_network:  # Default: auto-managed number of layers
                n_4x4_ch = 512 if self.spectrogram_channels == 1 else 768
            else:
                n_4x4_ch = 1800  # Forced number of layers, for some very specific experiments only
            self.features_mixer_cnn \
                = nn.Sequential(layer.Conv2D(256*self.spectrogram_channels, n_4x4_ch, [4, 4], [2, 2], 2, [1, 1],
                                             activation=nn.LeakyReLU(0.1), name_prefix='enc7'),
                                layer.Conv2D(n_4x4_ch, self.mixer_1x1conv_ch,
                                             [1, 1], [1, 1], 0, [1, 1],
                                             activation=nn.LeakyReLU(0.1), name_prefix='enc8', batch_norm=None)
                                )
        # - - - - - 3) MLP for extracting properly-sized latent vector - - - - -
        # Automatic CNN output tensor size inference
        with torch.no_grad():
            single_element_input_tensor_size = list(input_tensor_size)
            single_element_input_tensor_size[0] = 1  # single-element batch
            dummy_spectrogram = torch.zeros(single_element_input_tensor_size)
            self.cnn_out_size = self._forward_cnns(dummy_spectrogram).size()
        cnn_out_items = self.cnn_out_size[1] * self.cnn_out_size[2] * self.cnn_out_size[3]
        # No activation - outputs are latent mu/logvar
        if 'wavenet_baseline' in self.architecture\
                or 'speccnn8l1' in self.architecture:  # (not an MLP...) much is done in the CNN
            # TODO batch-norm here to compensate for unregularized z0 of a flow-based latent space (replace 0.1 Dkl)
            #    add corresponding ctor argument (build with bn=True if using flow-based latent space)
            # TODO remove this dropout?
            self.mlp = nn.Sequential(nn.Dropout(self.fc_dropout), nn.Linear(cnn_out_items, 2 * self.dim_z))
            if output_bn:
                self.mlp.add_module('lat_in_regularization', nn.BatchNorm1d(2 * self.dim_z))
        elif self.architecture == 'flow_synth':
            self.mlp = nn.Sequential(nn.Linear(cnn_out_items, 1024), nn.ReLU(),  # TODO dropouts
                                     nn.Linear(1024, 1024), nn.ReLU(),
                                     nn.Linear(1024, 2 * self.dim_z))
        else:
            raise NotImplementedError("Architecture '{}' not available".format(self.architecture))

    def _forward_cnns(self, x_spectrograms):
        # apply main cnn multiple times
        single_channel_cnn_out = [self.single_ch_cnn(torch.unsqueeze(x_spectrograms[:, ch, :, :], dim=1))
                                  for ch in range(self.spectrogram_channels)]
        # Then mix features from different input channels - and flatten the result
        return self.features_mixer_cnn(torch.cat(single_channel_cnn_out, dim=1))

    def forward(self, x_spectrograms):
        n_minibatch = x_spectrograms.size()[0]
        cnn_out = self._forward_cnns(x_spectrograms).view(n_minibatch, -1)  # 2nd dim automatically inferred
        # print("Forward CNN out size = {}".format(cnn_out.size()))
        z_mu_logvar = self.mlp(cnn_out)
        # Last dim contains a latent proba distribution value, last-1 dim is 2 (to retrieve mu or logs sigma2)
        return torch.reshape(z_mu_logvar, (n_minibatch, 2, self.dim_z))


class SpectrogramCNN(nn.Module):
    """ A encoder CNN network for spectrogram input """

    # TODO Option to enable res skip connections
    # TODO Option to choose activation function
    def __init__(self, architecture, last_layers_to_remove=0):
        """
        Automatically defines an autoencoder given the specified architecture

        :param last_layers_to_remove: Number of deepest conv layers to omit in this module (they will be added in
            the owner of this pure-CNN module).
        """
        super().__init__()
        self.architecture = architecture
        if last_layers_to_remove > 0:
            assert self.architecture == 'speccnn8l1_bn'  # Only this arch is fully-supported at the moment

        if self.architecture == 'wavenet_baseline'\
           or self.architecture == 'wavenet_baseline_lighter':  # this encoder is quite light already
            # TODO adapt to smaller spectrograms
            ''' Based on strided convolutions - no max pool (reduces the total amount of
             conv operations).  https://arxiv.org/abs/1704.01279
             No dilation: the receptive field in enlarged through a larger number
             of layers. 
             Layer 8 has a lower time-stride (better time resolution).
             Size of layer 9 (1024 ch) corresponds the wavenet time-encoder.
             
             Issue: when using the paper's FFT size and hop, layers 8 and 9 seem less useful. The image size
              at this depth is < kernel size (much of the 4x4 kernel convolves with zeros) '''
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

        elif self.architecture == 'wavenet_baseline_shallow':
            """ Inspired from wavenet_baseline, minus the two last layer, with less channels """
            self.enc_nn = nn.Sequential(layer.Conv2D(1, 8, [5, 5], [2, 2], 2, [1, 1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc1'),
                                        layer.Conv2D(8, 16, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc2'),
                                        layer.Conv2D(16, 32, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc3'),
                                        layer.Conv2D(32, 64, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc4'),
                                        layer.Conv2D(64, 128, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc5'),
                                        layer.Conv2D(128, 256, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc6'),
                                        layer.Conv2D(256, 512, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc7'),
                                        layer.Conv2D(512, 1024, [1, 1], [1, 1], 0, [1, 1],
                                                     activation=nn.LeakyReLU(0.1), name_prefix='enc8'),
                                        )

        elif self.architecture == 'flow_synth':
            # spectrogram (257, 347):   7.7 GB (RAM), 1.4 GMultAdd (batch 256) (inc. linear layers)
            ''' https://acids-ircam.github.io/flow_synthesizer/#models-details
            Based on strided convolutions and dilation to quickly enlarge the receptive field.
            Paper says: "5 layers with 128 channels of strided dilated 2-D convolutions with kernel
            size 7, stride 2 and an exponential dilation factor of 2l (starting at l=0) with batch
            normalization and ELU activation." Code from their git repo:
            dil = ((args.dilation == 3) and (2 ** l) or args.dilation)
            pad = 3 * (dil + 1)
            
            Potential issue: this dilation is extremely big for deep layers 4 and 5. Dilated kernel is applied
            mostly on zero-padded values. We should either stride-conv or 2^l dilate, but not both '''
            n_lay = 64  # 128/2 for paper's comparisons consistency. Could be larger
            self.enc_nn = nn.Sequential(layer.Conv2D(1, n_lay, [7,7], [2,2], 3, [1,1],
                                                     activation=nn.ELU(), name_prefix='enc1'),
                                        layer.Conv2D(n_lay, n_lay, [7, 7], [2, 2], 3, [2, 2],
                                                     activation=nn.ELU(), name_prefix='enc2'),
                                        layer.Conv2D(n_lay, n_lay, [7, 7], [2, 2], 3, [2, 2],
                                                     activation=nn.ELU(), name_prefix='enc3'),
                                        layer.Conv2D(n_lay, n_lay, [7, 7], [2, 2], 3, [2, 2],
                                                     activation=nn.ELU(), name_prefix='enc4'),
                                        layer.Conv2D(n_lay, n_lay, [7, 7], [2, 2], 3, [2, 2],
                                                     activation=nn.ELU(), name_prefix='enc5'))

        elif self.architecture == 'speccnn8l1':  # 1.7 GB (RAM) ; 0.12 GMultAdd  (batch 256)
            ''' Inspired by the wavenet baseline spectral autoencoder, but all sizes drastically reduced.
            Where to use BN?
            'Super-Resolution GAN' generator does not use BN in the first and last conv layers.'''
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            self.enc_nn = nn.Sequential(layer.Conv2D(1, 8, [5, 5], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc1'),
                                        layer.Conv2D(8, 16, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc2'),
                                        layer.Conv2D(16, 32, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc3'),
                                        layer.Conv2D(32, 64, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc4'),
                                        layer.Conv2D(64, 128, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc5'),
                                        layer.Conv2D(128, 256, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc6'),
                                        layer.Conv2D(256, 512, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc7'),
                                        layer.Conv2D(512, 1024, [1, 1], [1, 1], 0, [1, 1],
                                                     activation=act(act_p), name_prefix='enc8'),
                                        )
            # TODO le même mais avec des res-blocks add (avg-pool?)
            # TODO le même mais + profond (couches sans stride)
            # TODO le même mais + profond, en remplacer chaque conv 2d par un res-block 2 couches

        elif self.architecture == 'speccnn8l1_bn':  # 1.7 GB (RAM) ; 0.12 GMultAdd  (batch 256)
            ''' Where to use BN? 'ESRGAN' generator does not use BN in the first and last conv layers.
            DCGAN: no BN on discriminator in out generator out.
            Our experiments show: much more stable latent loss with no BNbefore the FC that regresses mu/logvar,
            consistent training runs 
            TODO try BN before act (see DCGAN arch) '''
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            self.enc_nn = nn.Sequential(layer.Conv2D(1, 8, [5, 5], [2, 2], 2, [1, 1], batch_norm=None,
                                                     activation=act(act_p), name_prefix='enc1'),
                                        layer.Conv2D(8, 16, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc2'),
                                        layer.Conv2D(16, 32, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc3'),
                                        layer.Conv2D(32, 64, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc4'),
                                        layer.Conv2D(64, 128, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc5'),
                                        layer.Conv2D(128, 256, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc6')
                                        )
            if last_layers_to_remove <= 1:
                self.enc_nn.add_module('4x4conv', layer.Conv2D(256, 512, [4, 4], [2, 2], 2, [1, 1],
                                                               activation=act(act_p), name_prefix='enc7'))
            if last_layers_to_remove == 0:
                self.enc_nn.add_module('1x1conv', layer.Conv2D(512, 1024, [1, 1], [1, 1], 0, [1, 1], batch_norm=None,
                                                               activation=act(act_p), name_prefix='enc8'))
        elif self.architecture == 'speccnn8l1_2':  # 5.8 GB (RAM) ; 0.65 GMultAdd  (batch 256)
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            self.enc_nn = nn.Sequential(layer.Conv2D(1, 32, [5, 5], [2, 2], 2, [1, 1], batch_norm=None,
                                                     activation=act(act_p), name_prefix='enc1'),
                                        layer.Conv2D(32, 64, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc2'),
                                        layer.Conv2D(64, 128, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc3'),
                                        layer.Conv2D(128, 128, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc4'),
                                        layer.Conv2D(128, 256, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc5'),
                                        layer.Conv2D(256, 256, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc6'),
                                        layer.Conv2D(256, 512, [4, 4], [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc7'),
                                        layer.Conv2D(512, 1024, [1, 1], [1, 1], 0, [1, 1], batch_norm=None,
                                                     activation=act(act_p), name_prefix='enc8'),
                                        )
        elif self.architecture == 'speccnn8l1_3':  # XXX GB (RAM) ; XXX GMultAdd  (batch 256)
            ''' speeccnn8l1_bn with bigger conv kernels '''
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            ker = [5, 5]  # TODO try bigger 1st ker?
            self.enc_nn = nn.Sequential(layer.Conv2D(1, 8, [5, 5], [2, 2], 2, [1, 1], batch_norm=None,
                                                     activation=act(act_p), name_prefix='enc1'),
                                        layer.Conv2D(8, 16, ker, [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc2'),
                                        layer.Conv2D(16, 32, ker, [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc3'),
                                        layer.Conv2D(32, 64, ker, [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc4'),
                                        layer.Conv2D(64, 128, ker, [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc5'),
                                        layer.Conv2D(128, 256, ker, [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc6'),
                                        layer.Conv2D(256, 512, ker, [2, 2], 2, [1, 1],
                                                     activation=act(act_p), name_prefix='enc7'),
                                        layer.Conv2D(512, 1024, [1, 1], [1, 1], 0, [1, 1], batch_norm=None,
                                                     activation=act(act_p), name_prefix='enc8'),
                                        )
        else:
            raise NotImplementedError("Architecture '{}' not available".format(self.architecture))

    def forward(self, x_spectrogram):
        return self.enc_nn(x_spectrogram)


if __name__ == "__main__":

    enc = SpectrogramEncoder('wavenet_baseline_reduced')
    print(enc)

    if False:
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


