"""
Easy-to-use metrics classes
"""

from collections import deque

import numpy as np

import torch


class BufferedMetric:
    """ Can store a limited number of metric values in order to get smoothed estimate of the metric. """
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.buffer = deque()

    def append(self, value):
        if isinstance(value, torch.Tensor):
            self.buffer.append(value.item())
        else:
            self.buffer.append(value)
        if len(self.buffer) > self.buffer_size:
            self.buffer.popleft()

    def mean(self):
        if len(self.buffer) == 0:
            raise ValueError()
        return np.asarray(self.buffer).mean()


class SimpleMetric:
    """ A very simple class for storing a metric, which provides EpochMetric-compatible methods """
    def __init__(self, value):
        self.v = value

    def on_new_epoch(self):
        return None

    def mean(self):
        return self.v


class EpochMetric:
    """ Can store mini-batch metric values in order to compute an epoch-averaged metric. """
    def __init__(self, normalized_losses=True):
        """
        :param normalized_losses: If False, the mini-batch size must be given when data is appended
        """
        # :param epoch_end_metric: If given, this class will append end-of-epoch values to this BufferedMetric instance
        self.normalized_losses = normalized_losses
        self.buffer = list()

    def on_new_epoch(self):
        self.buffer = list()

    def append(self, value, minibatch_size=-1):
        if minibatch_size <= 0:
            assert self.normalized_losses is True
        if isinstance(value, torch.Tensor):
            self.buffer.append(value.item())
        else:
            self.buffer.append(value)
            # TODO tester ça

    def mean(self):
        if len(self.buffer) == 0:
            raise ValueError()
        return np.asarray(self.buffer).mean()
