#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# hyperparameters optimization
# logs\train\multiruns\2024-09-11_14-58-53
python src/train.py model=unet hparams_search=unet logger=tensorboard
# tensorboard --logdir=logs/train/multiruns
# python src/eval.py model=unet logger=csv

# evaluation model with Defile values only
# logs\train\runs\2024-09-12_09-55-27
python src/train.py data=defile_only model=unet model.net.nb_input_features_hourly=20 model.net.nb_input_features_daily=7 logger=csv

# evaluation model with Defile data and daily at further sites
# logs\train\runs\2024-09-12_10-17-30
python src/train.py data=defile_hourly_only model=unet model.net.nb_input_features_hourly=20 model.net.nb_input_features_daily=22 logger=csv


python src/predict.py data=defile model=unet
