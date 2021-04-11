
"""
Script for cleaning the logs of a learning run, as described in the __main__ section

Useful to fully-clean Tensorboard logs:
- run this script
- refresh tensorboard
- tensorboard is ready for a new run with identical name
"""

# TODO command line arguments

from pathlib import Path

import config
import logs.logger
from utils.config import _Config


if __name__ == "__main__":

    if False:  # True to erase config from config.py, False to erase custom config given below
        model_config = config.model
    else:
        model_config = _Config
        model_config.logs_root_dir = "saved"
        # = = = = = Insert here model and run to be erased = = = = =
        model_config.name = 'MLPVAE3'
        model_config.run_name = '20b_dex3op_numonly_1midi_kf0'
        # = = = = = Insert here model and run to be erased = = = = =


    logs.logger.erase_run_data(Path(__file__).resolve().parent, model_config)

