
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
            'speccnn8l1'  # Custom 8-layer CNN + 1 linear very light architecture
            ]


class SpectrogramEncoder(nn.Module):
    """ Contains a spectrogram-input CNN and some MLP layers, and outputs the mu/logsigma2 values"""
    def __init__(self, architecture, dim_z, spectrogram_input_size):
        super().__init__()
        self.dim_z = dim_z  # Latent-vector size (2*dim_z encoded values - mu and log sigma 2)
        self.architecture = architecture
        self.cnn = SpectrogramCNN(self.architecture)
        # Automatic CNN output tensor size inference
        with torch.no_grad():
            dummy_spectrogram = torch.unsqueeze(torch.unsqueeze(torch.zeros(spectrogram_input_size), 0), 0)
            self.cnn_out_size = self.cnn(dummy_spectrogram).size()
        # MLP for extracting proper latent vector
        cnn_out_items = self.cnn_out_size[1] * self.cnn_out_size[2] * self.cnn_out_size[3]
        if 'wavenet_baseline' in self.architecture\
                or self.architecture == 'speccnn8l1':
            self.mlp = nn.Linear(cnn_out_items, 2 * self.dim_z)  # (not an MLP...) much is done in the CNN
        elif self.architecture == 'flow_synth':
            self.mlp = nn.Sequential(nn.Linear(cnn_out_items, 1024), nn.ReLU(),
                                     nn.Linear(1024, 1024), nn.ReLU(),
                                     nn.Linear(1024, 2 * self.dim_z))
        else:
            raise NotImplementedError("Architecture '{}' not available".format(self.architecture))

    def forward(self, x_spectrogram):
        n_minibatch = x_spectrogram.size()[0]
        cnn_out = self.cnn(x_spectrogram).view(n_minibatch, -1)  # 2nd dim automatically inferred
        # print("Forward CNN out size = {}".format(cnn_out.size()))
        z_mu_logsigma2 = self.mlp(cnn_out)
        # Last dim contains a latent proba distribution value, last-1 dim is 2 (to retrieve mu or log sigma2)
        return torch.reshape(z_mu_logsigma2, (n_minibatch, 2, self.dim_z))


class SpectrogramCNN(nn.Module):
    """ A encoder CNN network for spectrogram input """

    # TODO Option to enable res skip connections
    # TODO Option to choose activation function
    def __init__(self, architecture):
        """ Automatically defines an autoencoder given the specified architecture
        """
        super().__init__()
        self.architecture = architecture

        if self.architecture == 'wavenet_baseline'\
           or self.architecture == 'wavenet_baseline_lighter':  # this encoder is quite light already
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

        elif self.architecture == 'wavenet_baseline_shallow':  # Inspired from wavenet_baseline
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

        elif self.architecture == 'flow_synth':  # https://acids-ircam.github.io/flow_synthesizer/#models-details
            ''' Based on strided convolutions and dilation to quickly enlarge the receptive field.
            Paper says: "5 layers with 128 channels of strided dilated 2-D convolutions with kernel
            size 7, stride 2 and an exponential dilation factor of 2l (starting at l=0) with batch
            normalization and ELU activation."
            The padding is 3 * 2^l (not detailed in the paper).
            
            Potential issue: the dilation is extremely big for deep layers 4 and 5. Dilated kernel is applied
            mostly on zero-padded values. We should either stride-conv or 2^l dilate. Or maybe the
             dilation is not clearly explained in the paper (the dila) '''
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

        elif self.architecture == 'speccnn8l1':
            ''' Inspired by the wavenet baseline spectral autoencoder, but all sizes drastically reduced '''
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


