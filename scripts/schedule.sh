#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# hyperparameters optimization
python src/train.py hparams_search=defile_optuna logger=tensorboard
# tensorboard --logdir=logs/train/

# best model training
python src/train.py trainer.max_epochs=25 logger=tensorboard

# best model training
python src/eval.py ckpt_path=.logs/train/runs/2024-07-11_15-30-35/checkpoints/epoch_006.ckpt
