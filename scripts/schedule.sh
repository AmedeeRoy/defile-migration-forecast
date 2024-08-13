#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# hyperparameters optimization
python src/train.py model=default hparams_search=convnet logger=tensorboard
python src/train.py model=unet hparams_search=unet logger=tensorboard
python src/train.py model=transformer hparams_search=transformer logger=tensorboard
# tensorboard --logdir=logs/train/multiruns

# best model training
python src/train.py trainer.max_epochs=25 logger=tensorboard