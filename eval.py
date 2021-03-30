"""
Evaluation of trained models

TODO write doc
"""

import os
import os.path
from pathlib import Path
from datetime import datetime
from typing import Sequence
import multiprocessing
import copy

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

import data.build
import data.preset
import data.abstractbasedataset
import logs.logger
import logs.metrics
import model.build
import model.loss
import utils.audio
import utils.config
import synth.dexed


def evaluate_all_models(eval_config: utils.config.EvalConfig):
    """
    Evaluates all models whose names can be found in the given text file.

    :param eval_config:
    :return: TODO
    """
    # Retrieve the list of models to be evaluated
    root_path = Path(__file__).resolve().parent
    models_dirs_path = list()
    saved_folder_name = ("saved" if not eval_config.load_from_archives else "saved_archives")
    for model_name in eval_config.models_names:
        if eval_config.k_folds_count == 0:
            models_dirs_path.append(root_path.joinpath('{}/{}'.format(saved_folder_name, model_name)))
        else:  # Add k-folds if required
            for kf in range(eval_config.k_folds_count):
                models_dirs_path.append(root_path.joinpath('{}/{}_kf{}'.format(saved_folder_name, model_name, kf)))
    print("{} models found for evaluation".format(len(models_dirs_path)))

    # Single-model evaluation
    for i, model_dir_path in enumerate(models_dirs_path):
        print("================================================================")
        print("===================== Evaluation of model {}/{} ==================".format(i+1, len(models_dirs_path)))
        evaluate_model(model_dir_path, eval_config)


def get_eval_pickle_file_path(path_to_model_dir: Path, dataset_type: str, force_multi_note=False):
    return path_to_model_dir.joinpath('eval_{}{}.dataframe.pickle'
                                      .format(dataset_type, ('__MULTI_NOTE__' if force_multi_note else '')))


def evaluate_model(path_to_model_dir: Path, eval_config: utils.config.EvalConfig):
    """
    Loads a model from given directory (and its associated dataset) and performs a full evaluation
    TODO describe output
    """
    root_path = Path(__file__).resolve().parent
    t_start = datetime.now()

    # Special forced multi-note eval?
    if '__MULTI_NOTE__' in path_to_model_dir.name:
        forced_midi_notes = ((40, 85), (50, 85), (60, 42), (60, 85), (60, 127), (70, 85))
        # We'll load the original model - quite dirty path modification
        path_to_model_dir = Path(path_to_model_dir.__str__().replace('__MULTI_NOTE__', ''))
        if eval_config.verbosity >= 1:
            print("[eval.py] __MULTI_NOTE__ special evaluation")
    else:
        forced_midi_notes = None

    # Reload model and train config
    model_config, train_config = utils.config.get_config_from_file(path_to_model_dir.joinpath('config.json'))
    if eval_config.load_from_archives:
        model_config.logs_root_dir = "saved_archives"
    # Eval file to be created
    eval_pickle_file_path = get_eval_pickle_file_path(path_to_model_dir, eval_config.dataset,
                                                      force_multi_note=(forced_midi_notes is not None))
    # Return now if eval already exists, and should not be overridden
    if os.path.exists(eval_pickle_file_path):
        if not eval_config.override_previous_eval:
            if eval_config.verbosity >= 1:
                print("Evaluation file '{}' already exists. Skipping (override_previous_eval={})"
                      .format(eval_pickle_file_path, eval_config.override_previous_eval))
            return

    # Reload the corresponding dataset, dataloaders and models
    train_config.verbosity = 1
    train_config.minibatch_size = eval_config.minibatch_size  # Will setup dataloaders as requested
    # increase dataset size for multi-note eval or single-note trained models
    if forced_midi_notes is not None:
        model_config.midi_notes = forced_midi_notes
        model_config.increased_dataset_size = True
        if eval_config.verbosity >= 1:
            print("[eval.py] __MULTI_NOTE__: forced-increased dataset size (single MIDI -> multiple MIDI)")
    dataset = data.build.get_dataset(model_config, train_config)
    # dataloader is a dict of 3 subsets dataloaders ('train', 'validation' and 'test')
    dataloader, sub_datasets_lengths = data.build.get_split_dataloaders(train_config, dataset)
    # Rebuild model from last saved checkpoint (default: if trained on GPU, would be loaded on GPU)
    device = torch.device(eval_config.device)
    checkpoint = logs.logger.get_model_last_checkpoint(root_path, model_config, device=device)
    _, _, _, extended_ae_model = model.build.build_extended_ae_model(model_config, train_config,
                                                                     dataset.preset_indexes_helper)
    extended_ae_model.load_state_dict(checkpoint['ae_model_state_dict'])
    extended_ae_model = extended_ae_model.to(device).eval()
    ae_model, reg_model = extended_ae_model.ae_model, extended_ae_model.reg_model
    torch.set_grad_enabled(False)
    # Eval midi notes
    eval_midi_notes = (dataset.midi_notes if forced_midi_notes is None else forced_midi_notes)
    if eval_config.verbosity >= 1:
        print("Evaluation will be performed on {} MIDI note(s): {}".format(len(eval_midi_notes), eval_midi_notes))


    # = = = = = 0) Structures and Criteria for evaluation metrics = = = = =
    # Empty dicts (one dict per preset), eventually converted to a pandas dataframe
    eval_metrics = list()  # list of dicts
    preset_UIDs = list()
    synth_params_GT = list()
    synth_params_inferred = list()
    # Parameters criteria
    controls_num_mse_criterion = model.loss.QuantizedNumericalParamsLoss(dataset.preset_indexes_helper,
                                                                         numerical_loss=nn.MSELoss(reduction='mean'))
    controls_num_mae_criterion = model.loss.QuantizedNumericalParamsLoss(dataset.preset_indexes_helper,
                                                                         numerical_loss=nn.L1Loss(reduction='mean'))
    controls_accuracy_criterion = model.loss.CategoricalParamsAccuracy(dataset.preset_indexes_helper,
                                                                       reduce=True, percentage_output=True)
    # Controls related to MIDI key and velocity (to compare single- and multi-channel spectrograms models)
    if dataset.synth_name.lower() == "dexed":
        dynamic_vst_controls_indexes = synth.dexed.Dexed.get_midi_key_related_param_indexes()
    else:
        raise NotImplementedError("")
    dynamic_controls_num_mae_crit = \
        model.loss.QuantizedNumericalParamsLoss(dataset.preset_indexes_helper,
                                                numerical_loss=nn.L1Loss(reduction='mean'),
                                                limited_vst_params_indexes=dynamic_vst_controls_indexes)
    dynamic_controls_acc_crit = \
        model.loss.CategoricalParamsAccuracy(dataset.preset_indexes_helper, reduce=True, percentage_output=True,
                                             limited_vst_params_indexes=dynamic_vst_controls_indexes)
    # correlation results - will be written to a separate pickle file (not averaged, for detailed study)
    z0_metric = logs.metrics.CorrelationMetric(model_config.dim_z, sub_datasets_lengths[eval_config.dataset])
    zK_metric = logs.metrics.CorrelationMetric(model_config.dim_z, sub_datasets_lengths[eval_config.dataset])


    # = = = = = 1) Infer all preset parameters = = = = =
    assert eval_config.minibatch_size == 1  # Required for per-preset metrics
    # This dataset's might contains multiple MIDI notes for single-note models, if forced_midi_notes is True
    for i, sample in enumerate(dataloader[eval_config.dataset]):
        x_in, v_in, sample_info = sample[0].to(device), sample[1].to(device), sample[2].to(device)
        ae_out = ae_model(x_in, sample_info)  # Spectral VAE - tuple output
        z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac, x_out = ae_out
        z0_metric.append_batch(z_0_sampled)
        zK_metric.append_batch(z_K_sampled)
        v_out = reg_model(z_K_sampled)
        # Parameters inference metrics
        preset_UIDs.append(sample_info[0, 0].item())
        eval_metrics.append(dict())
        eval_metrics[-1]['preset_UID'] = sample_info[0, 0].item()
        eval_metrics[-1]['num_controls_MSEQ'] = controls_num_mse_criterion(v_out, v_in).item()
        eval_metrics[-1]['num_controls_MAEQ'] = controls_num_mae_criterion(v_out, v_in).item()
        eval_metrics[-1]['cat_controls_acc'] = controls_accuracy_criterion(v_out, v_in)
        eval_metrics[-1]['num_dyn_cont_MAEQ'] = dynamic_controls_num_mae_crit(v_out, v_in).item()
        eval_metrics[-1]['cat_dyn_cont_acc'] = dynamic_controls_acc_crit(v_out, v_in)
        # Compute corresponding flexible presets instances
        in_presets_instance = data.preset.DexedPresetsParams(learnable_presets=v_in, dataset=dataset)
        out_presets_instance = data.preset.DexedPresetsParams(learnable_presets=v_out, dataset=dataset)
        # VST-compatible full presets (1-element batch of presets)
        synth_params_GT.append(in_presets_instance.get_full()[0, :].cpu().numpy())
        synth_params_inferred.append(out_presets_instance.get_full()[0, :].cpu().numpy())

    # Numpy matrix of preset values. Reconstructed spectrograms are not stored
    synth_params_GT, synth_params_inferred = np.asarray(synth_params_GT), np.asarray(synth_params_inferred)
    preset_UIDs = np.asarray(preset_UIDs)


    # = = = = = 2) Evaluate audio from inferred synth parameters = = = = =
    num_workers = int(np.round(os.cpu_count() * eval_config.multiprocess_cores_ratio))
    preset_UIDs_split = np.array_split(preset_UIDs, num_workers, axis=0)
    synth_params_GT_split = np.array_split(synth_params_GT, num_workers, axis=0)
    synth_params_inferred_split = np.array_split(synth_params_inferred, num_workers, axis=0)
    workers_data = [(dataset, eval_midi_notes,
                     preset_UIDs_split[i], synth_params_GT_split[i], synth_params_inferred_split[i])
                    for i in range(num_workers)]
    # Multi-processing is absolutely necessary
    with multiprocessing.Pool(num_workers) as p:
        audio_errors_split = p.map(_measure_audio_errors_worker, workers_data)
    audio_errors = dict()
    for error_name in audio_errors_split[0]:
        audio_errors[error_name] = np.hstack([audio_errors_split[i][error_name]
                                              for i in range(len(audio_errors_split))])


    # = = = = = 3) Concatenate results into a dataframe = = = = =
    eval_df = pd.DataFrame(eval_metrics)
    # append audio errors
    for error_name, err in audio_errors.items():
        eval_df[error_name] = err
    # multi-note case: average results with the same preset UID (Python set prevents duplicates)
    # This also sorts the dataframe presets UIDs and will done for all evaluations (sub-optimal but small data structs)
    preset_UIDs_no_duplicates = list(set(eval_df['preset_UID'].values))
    preset_UIDs_no_duplicates.sort()
    eval_metrics_no_duplicates = list()  # Will eventually be a dataframe
    # We use the original list to build a new dataframe
    for preset_UID in preset_UIDs_no_duplicates:
        eval_metrics_no_duplicates.append(dict())
        eval_sub_df = eval_df.loc[eval_df['preset_UID'] == preset_UID]
        eval_metrics_no_duplicates[-1]['preset_UID'] = preset_UID
        for col in eval_sub_df:  # Average all metrics
            if col != 'preset_UID':
                eval_metrics_no_duplicates[-1][col] = eval_sub_df[col].mean()
    eval_df = pd.DataFrame(eval_metrics_no_duplicates)


    # = = = = = 4) Write eval files = = = = =
    # Main dataframe
    eval_df.to_pickle(eval_pickle_file_path)
    # Additional numpy files
    try:
        os.mkdir(path_to_model_dir.joinpath('eval_files'))
    except FileExistsError:
        pass  # Directory was already created
    spearman_r, spearman_pvalues = z0_metric.get_spearman_corr_and_p_values()
    np.save(path_to_model_dir.joinpath('eval_files/z0_spearman_r__{}.npy'.format(eval_config.dataset)), spearman_r)
    np.save(path_to_model_dir.joinpath('eval_files/z0_spearman_pvalues__{}.npy'.format(eval_config.dataset)),
            spearman_pvalues)
    spearman_r, spearman_pvalues = zK_metric.get_spearman_corr_and_p_values()
    np.save(path_to_model_dir.joinpath('eval_files/zK_spearman_r__{}.npy'.format(eval_config.dataset)), spearman_r)
    np.save(path_to_model_dir.joinpath('eval_files/zK_spearman_pvalues__{}.npy'.format(eval_config.dataset)),
            spearman_pvalues)
    # End of eval
    if eval_config.verbosity >= 1:
        print("Finished evaluation ({}) in {:.1f}s".format(eval_pickle_file_path,
                                                           (datetime.now() - t_start).total_seconds()))


def _measure_audio_errors_worker(worker_args):
    return _measure_audio_errors(*worker_args)


def _measure_audio_errors(dataset: data.abstractbasedataset.PresetDataset, midi_notes,
                          preset_UIDs: Sequence, synth_params_GT: np.ndarray, synth_params_inferred: np.ndarray):
    # Dict of per-UID errors (if multiple notes: note-averaged values)
    errors = {'spec_mae': list(), 'spec_sc': list(), 'mfcc13_mae': list(), 'mfcc40_mae': list()}
    for idx, preset_UID in enumerate(preset_UIDs):
        mae, sc, mfcc13_mae, mfcc40_mae = list(), list(), list(), list()  # Per-note errors (might be 1-element lists)
        for midi_pitch, midi_velocity in midi_notes:  # Possible multi-note evaluation
            x_wav_original, _ = dataset.get_wav_file(preset_UID, midi_pitch, midi_velocity)  # Pre-rendered file
            x_wav_inferred, _ = dataset._render_audio(synth_params_inferred[idx], midi_pitch, midi_velocity)
            similarity_eval = utils.audio.SimilarityEvaluator((x_wav_original, x_wav_inferred))
            mae.append(similarity_eval.get_mae_log_stft(return_spectrograms=False))
            sc.append(similarity_eval.get_spectral_convergence(return_spectrograms=False))
            mfcc13_mae.append(similarity_eval.get_mae_mfcc(return_mfccs=False, n_mfcc=13))
            mfcc40_mae.append(similarity_eval.get_mae_mfcc(return_mfccs=False, n_mfcc=40))
        # Average errors over all played MIDI notes
        errors['spec_mae'].append(np.mean(mae))
        errors['spec_sc'].append(np.mean(sc))
        errors['mfcc13_mae'].append(np.mean(mfcc13_mae))
        errors['mfcc40_mae'].append(np.mean(mfcc40_mae))
    for error_name in errors:
        errors[error_name] = np.asarray(errors[error_name])
    return errors


if __name__ == "__main__":
    import evalconfig
    eval_config = evalconfig.eval

    print("Starting models evaluation using configuration from evalconfig.py, using '{}' dataset"
          .format(eval_config.dataset))
    evaluate_all_models(eval_config)


