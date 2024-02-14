#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# hyperparameters optimization
python src/train.py hparams_search=defile_optuna

# best model training
python src\train.py trainer.max_epochs=25 logger=tensorboard

# best model training
python src\eval.py