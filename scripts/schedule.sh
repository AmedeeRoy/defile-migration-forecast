#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# hyperparameters optimization
python src/train.py hparams_search=defile_optuna logger=tensorboard
# tensorboard --logdir=logs/train/mutliruns/ &

# best model training
python src/train.py trainer.max_epochs=25 logger=tensorboard
# tensorboard --logdir=logs/train/runs/2024-07-10_13-06-57 &

# best model training
python src/eval.py
