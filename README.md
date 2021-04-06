# preset-gen-vae

## Data

* ```./synth/``` Dexed presets SQLite main database, Python modules, and pre-rendered .pickle/.txt/.wav files
* ```./data/``` PyTorch dataset and pre-computed spectrogram stats

## Saved models and logs
* ```./saved/runs/model_name/run_name``` contains Tensorboard data for all models and learning runs. This
  hierarchical structure allows comparing models and/or runs of a given model 
* ```./saved/model_name/run_name/config.json``` stores the full config (model, hyperparams, etc...) of a given run  
* ```./saved/model_name/run_name/models/```  stores trained models (sorted by epoch)
