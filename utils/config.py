
import json
import collections


# Could (should) be improved...
class _Config(object):
    pass


class EvalConfig:
    def __init__(self):
        self.start_datetime = ''
        self.models_names = ['']
        self.override_previous_eval = False
        self.device = ''
        self.verbosity = 0
        self.dataset = ''
        self.k_folds_count = 0
        self.minibatch_size = 0
        self.load_from_archives = False
        self.multiprocess_cores_ratio = 0.5


class LoadedRunConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_config_from_file(absolute_file_path):
    """ Returns the model_config and train_config from a config.json saved run file,
    which contains the same values as the original config.py file that was used during trainin. """
    with open(absolute_file_path) as f:
        config_json = json.load(f)
    # All lists (default json parse) must be converted to tuples
    for main_k in config_json:
        for k, v in config_json[main_k].items():
            if isinstance(v, list):
                config_json[main_k][k] = tuple(v)
    # Transform dict to class attributes
    config_objects = {}
    for main_k in config_json:
        '''
        config_objects[main_k] = collections.namedtuple('LoadedConfig', config_json[main_k].keys())
        for k, v in config_json[main_k].items():
            config_objects[main_k].__dict__[k] = v
            '''
        config_objects[main_k] = LoadedRunConfig(**config_json[main_k])

    return config_objects['model'], config_objects['train']


if __name__ == "__main__":
    # config.json read test
    model_config, train_config = get_config_from_file("/home/gwendal/Jupyter/nn-synth-interp/saved/BasicVAE/11_tanh_output/config.json")
    pass
