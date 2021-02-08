# nn-synth-interp

## Data

* ```./synth/``` Dexed presets SQLite main database, Python modules, and pre-rendered .pickle/.txt/.wav files
* ```./data/``` PyTorch dataset and pre-computed spectrogram stats

## Saved models and logs
* ```./saved/runs/model_name/run_name``` contains Tensorboard data for all models and learning runs. This
  hierarchical structure allows comparing models and/or runs of a given model 
* ```./saved/model_name/run_name/config.json``` stores the full config (model, hyperparams, etc...) of a given run  
* ```./saved/model_name/run_name/models/```  stores trained models (sorted by epoch)
* Same structure in ```./saved_archives``` (put all old/deprecated reference models in here)

## Branches

* ```main``` .ipynb Notebooks are to be used in Jupyter Lab running on the ML server, through the web interface
    * Excluded from PyCharm auto-deploy
* ```packages``` Pure Python code (in folders) modified by remote PyCharm only
