

import torch.nn as nn


class MSELoss(nn.MSELoss):
    """ Very basic wrapper with an added get_short_name method """
    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)

    @staticmethod
    def get_short_name(self):
        return 'MSE'

# TODO Spectral Convergence

# TODO D_KL formule directe en connaissant mu/sigma

# TODO MMD