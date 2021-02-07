"""
Script that can be edited to configure and run a queue of training runs.
Must be run as main

See the actual training function in train.py
"""


import importlib  # to reload config.py between each run
import numpy as np

import train


# = = = = = = = = = = config.py modifications = = = = = = = = = =
model_config_mods, train_config_mods = list(), list()
"""
Please write two lists of dicts, such that:
- (model|train)_config_mods contains the modifications applied to config.model and config.train, resp.
- each list index corresponds to a training run
- each dict key corresponds to an attribute of config.model.* or config.train.*. Empty dict to indicate
that no config modification should be performed
"""
# Run 0
model_config_mods.append({})
train_config_mods.append({})
# Run 1
model_config_mods.append({'run_name': '04_8l1_plateau10_reconsloss'})
train_config_mods.append({'scheduler_loss': 'ReconsLoss'})

# = = = = = = = = = = end of config.py modifications = = = = = = = = = =


if __name__ == "__main__":

    assert len(model_config_mods) == len(train_config_mods)

    for run_index in range(len(model_config_mods)):
        # Force config reload
        import config
        importlib.reload(config)

        print("================================================================")
        print("=============== Enqueued Training Run {}/{} starts ==============="
              .format(run_index+1, len(model_config_mods)))

        # Direct dirty modification of config.py module attributes
        for k, v in model_config_mods[run_index].items():
            config.model.__dict__[k] = v
        for k, v in train_config_mods[run_index].items():
            config.train.__dict__[k] = v

        # This dynamically modified config.py will be used by train.py
        train.train_config()

        print("=============== Enqueued Training Run {}/{} has finished ==============="
              .format(run_index+1, len(model_config_mods)))
        print("======================================================================")
