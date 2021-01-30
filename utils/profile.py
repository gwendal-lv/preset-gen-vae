
import contextlib
from torch.autograd import profiler


def get_optional_profiler(profiler_args):
    """ Can return an ActualProfiler, or a NoProfiler that really does nothing when disabled.

    PyTorch might implement this in some way, but record_function(..) in profiler.py
    actually creates an 'handle-tensor' """
    if profiler_args['enabled']:
        return ActualProfiler(profiler_args)
    else:
        return NoProfiler(profiler_args)


class ActualProfiler(profiler.profile):
    def __init__(self, profiler_args):
        """ Creates a PyTorch profile class instance, with a few added methods """
        super().__init__(**profiler_args)
        assert profiler_args['enabled']  # This class is not intended to be used as a disabled profiler

    def record_function(self, name):
        """ Function for compatibility with NoProfiler (not static to prevent static call) """
        return profiler.record_function(name)


class NoProfiler(contextlib.nullcontext):
    """ Class that actualy does nothing, but which can be called as an ActualProfiler """
    def __init__(self, profiler_args=None):
        super().__init__()
        if profiler_args is not None:
            assert not profiler_args['enabled']  # This class should be used for disabled profiling use cases
        self.enabled = False

    def record_function(self, name):
        return contextlib.nullcontext()
