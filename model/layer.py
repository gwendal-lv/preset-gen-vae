
"""
Defines some basic layer Classes to be integrated into bigger networks
"""

import torch
import torch.nn as nn


class Conv2D(nn.Sequential):
    """ A basic conv layer with activation and batch-norm """
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation,
                 padding_mode='zeros', activation=nn.ReLU, name_prefix='', batch_norm_first=False):
        super().__init__()
        self.add_module(name_prefix + 'conv', nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                                                        padding, dilation, padding_mode=padding_mode))
        bn = nn.BatchNorm2d(out_ch)
        if batch_norm_first:
            self.add_module(name_prefix + 'bn', bn)
        self.add_module(name_prefix + 'act', activation)
        if not batch_norm_first:
            self.add_module(name_prefix + 'bn', bn)


class TConv2D(nn.Sequential):
    """ A basic Transposed conv layer with activation and batch-norm """
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, output_padding=0, dilation=1,
                 padding_mode='zeros', activation=nn.ReLU, name_prefix='', batch_norm_first=False):
        super().__init__()
        self.add_module(name_prefix + 'tconv', nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride,
                                                                  padding, output_padding,
                                                                  dilation=dilation, padding_mode=padding_mode))
        bn = nn.BatchNorm2d(out_ch)
        if batch_norm_first:
            self.add_module(name_prefix + 'bn', bn)
        self.add_module(name_prefix + 'act', activation)
        if not batch_norm_first:
            self.add_module(name_prefix + 'bn', bn)
