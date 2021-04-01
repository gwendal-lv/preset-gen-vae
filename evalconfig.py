"""
Allows easy modification of all configuration parameters required to perform a series of models evaluations.
This script is not intended to be run, it only describes parameters.
"""


import datetime
from utils.config import EvalConfig


eval = EvalConfig()  # (shadows unused built-in name)
eval.start_datetime = datetime.datetime.now().isoformat()

# Names must be include experiment folder and run name (_kf suffix must be omitted is all_k_folds is True)
eval.models_names = [  # - - - 16k samples dataset - - -
                     #'ExtVAE3/10_dex3op_numonly_1midi',  # Single-note models evaluated on 1 MIDI note
                     #'ExtVAE3/11_dex6op_numonly_1midi',
                     #'ExtVAE3/12_dex3op_vstcat_1midi',
                     #'ExtVAE3/13_dex6op_vstcat_1midi',
                     #'ExtVAE3/14_dex3op_all<=32_1midi',
                     #'ExtVAE3/14_dex3op_all<=32_1midi__MULTI_NOTE__',  # Special eval: forced multi-note eval
                     #'ExtVAE3/15_dex6op_all<=32_1midi',
                     #'ExtVAE3/15_dex6op_all<=32_1midi__MULTI_NOTE__',
                     #'MLPVAE/20_dex3op_numonly_1midi',
                     #'MLPVAE/21_dex6op_numonly_1midi',
                     #'MLPVAE/22_dex3op_vstcat_1midi',
                     #'MLPVAE/23_dex6op_vstcat_1midi',
                     #'MLPVAE/24_dex3op_all<=32_1midi',
                     #'MLPVAE/25_dex6op_all<=32_1midi',
                     #'FlVAE/34_dex3op_all<=32_6midi',  # Multi-note models evaluated on all learned notes
                     #'FlVAE/35_dex6op_all<=32_6midi',
                     #'FlVAE/44_dex3op_all<=32_6stack',
                     #'FlVAE/45_dex6op_all<=32_6stack',
                     # - - - 30k samples full dataset ('b' suffix means 'big') - - -
                     'FlVAE2/14b_dex3op_all<=32_1midi',
                     'FlVAE2/15b_dex6op_all<=32_1midi',
                     'FlVAE2/44b_dex3op_all<=32_6stack'
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
