# preset-gen-vae

## Introduction

This repository provides models and data to learn how to program a Dexed FM synthesizer (DX7 software clone) from an input sound.
Models based on Variational Autoencoders (VAE) and results are described in the [DAFx 2021 paper](https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in21_paper_7.pdf) and the [companion website](https://gwendal-lv.github.io/preset-gen-vae/).

Neural networks and training procedures can be configured using the ```config.py``` file.

## Project structure

### Data

* ```./synth/``` Dexed presets SQLite main database (> 30k presets), Python modules, and pre-rendered .pickle/.txt/.wav files
* ```./data/``` PyTorch dataset and pre-computed spectrogram stats

### Saved models and logs
* ```./saved/runs/model_name/run_name``` contains Tensorboard data for all models and learning runs
* ```./saved/model_name/run_name/config.json``` stores the full config (model, hyperparams, etc...) of a given run  
* ```./saved/model_name/run_name/models/```  stores trained models (sorted by epoch)

## Citation

If you use this work in your research, please cite:

```
@inproceedings{levaillant2021vaesynthprog,
	title        = {Improving Synthesizer Programming from Variational Autoencoders Latent Space},
	author       = {Le Vaillant, Gwendal and Dutoit, Thierry and Dekeyser, Sébastien},
	year         = 2021,
	month        = Sep,
	booktitle    = {Proceedings of the 24th International Conference on Digital Audio Effects (DAFx20in21)},
	location     = {Vienna, Austria}
}
```
