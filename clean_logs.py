
"""
Script for cleaning the logs of the learning run configured in config.py.

Useful to fully-clean Tensorboard logs:
- run this script
- refresh tensorboard
- tensorboard is ready for a new run with identical name
"""

# TODO command line arguments

from pathlib import Path

import config
import log.logger


if __name__ == "__main__":
    log.logger.erase_run_data(Path(__file__).resolve().parent, config.model)

