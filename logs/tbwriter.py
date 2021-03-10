
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from .metrics import BufferedMetric


class CorrectedSummaryWriter(SummaryWriter):
    """ SummaryWriter corrected to prevent extra runs to be created
    in Tensorboard when adding hparams.

    Original code in torch/utils/tensorboard.writer.py,
    modification by method overloading inspired by https://github.com/pytorch/pytorch/issues/32651 """

    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):
        assert run_name is None  # Disabled feature. Run name init by summary writer ctor

        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        # run_name argument is discarded and the writer itself is used (no extra writer instantiation)
        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)


class TensorboardSummaryWriter(CorrectedSummaryWriter):
    """ Tensorboard SummaryWriter with corrected add_hparams method
     and extra functionalities. """

    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix='',
                 model_config=None, train_config=None  # Added (actually mandatory) arguments
                 ):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        # Full-Config is required. Default constructor values allow to keep the same first constructor args
        self.model_config = model_config
        self.train_config = train_config
        self.resume_from_checkpoint = (train_config.start_epoch > 0)
        self.hyper_params = dict()
        # General and dataset hparams
        self.hyper_params['batchsz'] = self.train_config.minibatch_size
        self.hyper_params['kfold'] = self.train_config.current_k_fold
        self.hyper_params['wdecay'] = self.train_config.weight_decay
        self.hyper_params['fcdrop'] = self.train_config.fc_dropout
        self.hyper_params['synth'] = self.model_config.synth
        self.hyper_params['syntargs'] = self.model_config.synth_args_str
        self.hyper_params['catmodel'] = self.model_config.synth_vst_params_learned_as_categorical
        self.hyper_params['normloss'] = self.train_config.normalize_losses
        # Latent space hparams
        self.hyper_params['z_dim'] = self.model_config.dim_z
        # self.hyper_params['latloss'] = self.train_config.latent_loss
        self.hyper_params['controls'] = self.model_config.synth_params_count
        # Synth controls regression - not logged anymore (see model_config.synth_vst_params_learned_as_categorical)
        # self.hyper_params['contloss'] = self.model_config.controls_losses
        # Auto-Encoder hparams
        self.hyper_params['encarch'] = self.model_config.encoder_architecture
        # self.hyper_params['recloss'] = self.train_config.ae_reconstruction_loss
        self.hyper_params['mels'] = self.model_config.mel_bins
        self.hyper_params['mindB'] = self.model_config.spectrogram_min_dB
        self.hyper_params['melfmin'] = self.model_config.mel_f_limits[0]
        self.hyper_params['melfmax'] = self.model_config.mel_f_limits[1]
        # TODO hparam domain discrete

    def init_hparams_and_metrics(self, metrics):
        """ Hparams and Metric initialization. Will pass if training resumes from saved checkpoint.
        Hparams will be definitely set but metrics can be updated during training.

        :param metrics: Dict of BufferedMetric
        """
        if not self.resume_from_checkpoint:  # tensorboard init at epoch 0 only
            # Some processing on hparams can be done here... none at the moment
            self.update_metrics(metrics)

    def update_metrics(self, metrics):
        """ Updates Tensorboard metrics

        :param metrics: Dict of values and/or BufferedMetric instances
        :return: None
        """
        metrics_dict = dict()
        for k, metric in metrics.items():
            if isinstance(metrics[k], BufferedMetric):
                try:
                    metrics_dict[k] = metric.mean
                except ValueError:
                    metrics_dict[k] = 0  # TODO appropriate default metric value?
            else:
                metrics_dict[k] = metric
        self.add_hparams(self.hyper_params, metrics_dict, hparam_domain_discrete=None)

