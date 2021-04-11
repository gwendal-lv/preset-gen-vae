"""
Allows easy modification of all configuration parameters required to perform a series of models evaluations.
This script is not intended to be run, it only describes parameters.
"""


import datetime
from utils.config import EvalConfig


eval = EvalConfig()  # (shadows unused built-in name)
eval.start_datetime = datetime.datetime.now().isoformat()

# Names must be include experiment folder and run name (_kf suffix must be omitted is all_k_folds is True)
eval.models_names = [  # - - - 30k samples full dataset ('b' suffix means 'big') - - -
                     'FlVAE3/10b_dex3op_numonly_1midi',
                     'FlVAE3/11b_dex6op_numonly_1midi',
                     'FlVAE3/12b_dex3op_vstcat_1midi',
                     'FlVAE3/13b_dex6op_vstcat_1midi',
                     'FlVAE3/14b_dex3op_all<=32_1midi',
                     'FlVAE3/15b_dex6op_all<=32_1midi',
                     'MLPVAE3/20b_dex3op_numonly_1midi',
                     'MLPVAE3/21b_dex6op_numonly_1midi',
                     'MLPVAE3/22b_dex3op_vstcat_1midi',
                     'MLPVAE3/23b_dex6op_vstcat_1midi',
                     'MLPVAE3/24b_dex3op_all<=32_1midi',
                     'MLPVAE3/25b_dex6op_all<=32_1midi',
                     'FlVAE3/34b_dex3op_all<=32_6midi',
                     'FlVAE3/35b_dex6op_all<=32_6midi',
                     'FlVAE3/44b_dex3op_all<=32_6stack',
                     'FlVAE3/45b_dex6op_all<=32_6stack',
                     ]
eval.dataset = 'test'  # Do not use 'test' dataset during models development
eval.override_previous_eval = False  # If True, all models be re-evaluated (might be very long)
eval.k_folds_count = 5  # 0 means do not automatically process all k-folds trains

eval.minibatch_size = 1  # Reduced mini-batch size not to reserve too much GPU RAM. 1 <=> per-preset metrics
eval.device = 'cpu'
# Don't use too many cores, numpy uses multi-threaded MKL (in each process)
eval.multiprocess_cores_ratio = 0.1  # ratio of CPU cores to be used (if 1.0: use all os.cpu_count() cores)
eval.verbosity = 2
eval.load_from_archives = False  # Load from ./saved_archives instead of ./saved
