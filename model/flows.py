"""
Normalizing flows classes and utils.
Most flow transform are directly defined in models constructors (see VAE.py, regression.py)
"""

import torch
from torch import nn
from torch.nn import functional as F

from nflows.nn import nets as nets
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AdditiveCouplingTransform, AffineCouplingTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.normalization import BatchNorm



class InverseFlow(nn.Module):
    """
    CompositeTransform (nflows package) wrapper which reverses the flow .inverse and .forward methods.

    Useful when combined with DataParallel (which only provides .forward calls) or reverse the fast/slow
    properties of the forward/inverse calls of the original flow.
    """
    def __init__(self, flow: CompositeTransform):
        super().__init__()
        raise AssertionError("This class messes autograd graphs (or only pytorch summaries???) and will be removed")
        assert isinstance(flow, CompositeTransform)
        self.flow = flow

    def forward(self, z):
        return self.flow.inverse(z)

    def inverse(self, z):
        return self.flow(z)



class CustomRealNVP(CompositeTransform):
    """ A slightly modified version of the SimpleRealNVP from nflows,
     which is a CompositeTransform and not a full Flow with base distribution. """
    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        use_volume_preserving=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):

        if use_volume_preserving:
            coupling_constructor = AdditiveCouplingTransform
        else:
            coupling_constructor = AffineCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        use_dropout = True  # Quick and dirty: 'global' variable, as seen by the create_resnet function

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability if use_dropout else 0.0,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for l in range(num_layers):
            use_dropout = l < (num_layers-2)  # No dropout on the 2 last layers
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_resnet
            )
            layers.append(transform)
            mask *= -1  # Checkerboard masking inverse
            if batch_norm_between_layers and l < (num_layers-2):  # No batch norm on the last 2 layers
                layers.append(BatchNorm(features=features))

        super().__init__(layers)



if __name__ == "__main__":
    pass


