"""
Allows easy modification of all configuration parameters required to perform a series of models evaluations.
This script is not intended to be run, it only describes parameters.
"""


import datetime
from utils.config import EvalConfig


eval = EvalConfig()  # (shadows unused built-in name)
eval.start_datetime = datetime.datetime.now().isoformat()

# Names must be include experiment folder and run name (_kf suffix can be omitted is all_k_folds is True)
# ExtVAE3/00_*** had a 1024-ch mixer
#'ExtVAE3/00_dex3op_numonly_1midi',
#'ExtVAE3/01_dex6op_numonly_1midi',
#'ExtVAE3/02_dex3op_vstcat_1midi',
#'ExtVAE3/03_dex6op_vstcat_1midi',
eval.models_names = ['ExtVAE3/10_dex3op_numonly_1midi',
                     'ExtVAE3/11_dex6op_numonly_1midi',
                     'ExtVAE3/12_dex3op_vstcat_1midi',
                     'ExtVAE3/13_dex6op_vstcat_1midi',
                     'ExtVAE3/14_dex3op_all<=32_1midi',
                     'ExtVAE3/15_dex6op_all<=32_1midi',
                     ]
eval.dataset = 'test'  # Do not use 'test' dataset during models development
eval.override_previous_eval = False  # If True, all models be re-evaluated (might be very long)
eval.k_folds_count = 5  # 5  # 0 means do not automatically all k-folds trains

eval.minibatch_size = 1  # Reduced mini-batch size not to reserve too much GPU RAM. 1 <=> per-preset metrics
eval.device = 'cpu'
# Don't use too many cores, numpy uses multi-threaded MKL (in each process)
eval.multiprocess_cores_ratio = 0.1  # ratio of CPU cores to be used (if 1.0: use all os.cpu_count() cores)
eval.verbosity = 2
eval.load_from_archives = False  # Load from ./saved_archives instead of ./saved
