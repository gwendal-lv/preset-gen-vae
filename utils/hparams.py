

class LinearDynamicParam:
    """ Hyper-Parameter which is able to automatically increase or decrease at each epoch.
    It provides the same methods as a metric (see logs/metrics.py) and can be easily used with tensorboard. """
    def __init__(self, start_value, end_value, start_epoch=0, end_epoch=10, current_epoch=-1):
        self.current_epoch = current_epoch - 1  # This value will be incremented when epoch actually starts
        self.start_value = start_value
        self.end_value = end_value
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        assert self.end_epoch >= self.start_epoch

    def on_new_epoch(self):
        self.current_epoch += 1

    def get(self, current_epoch=None):
        """ Returns an interpolated value. Current epoch was automatically incremented by calling on_new_epoch()
        but can be passed as argument to this method. """
        if current_epoch is None:
            current_epoch = self.current_epoch
        else:
            self.current_epoch = current_epoch
        if current_epoch >= self.end_epoch:
            return self.end_value
        elif current_epoch <= self.start_epoch:
            return self.start_value
        else:
            offset_epochs = current_epoch - self.start_epoch
            return self.start_value + (self.end_value - self.start_value) * offset_epochs\
                                            / (self.end_epoch - self.start_epoch)

    @property
    def value(self):
        return self.get()


