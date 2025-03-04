#!/bin/bash

# TRAIN
# Main training with all data
python src/train.py

python src/train.py data="test"

# Alternative 1: evaluation model with Defile values only
python src/train.py experiment=defile_only data.species="Common Buzzard"

# Alternative 2: evaluation model with Defile data and daily at further sites
python src/train.py experiment=defile_hourly_only 


# PREDICT
python src/predict.py data.species="Common Buzzard"


# PRODUCTION

python src/train.py --multirun experiment=production data.species="Common Buzzard","Red Kite","Black Kite","European Honey-buzzard","Western Marsh Harrier","Eurasian Sparrowhawk","Eurasian Kestrel","Osprey","Eurasian Hobby","Hen Harrier","Merlin" logger=csv
python src/predict.py --multirun experiment=production data.species="Common Buzzard","Red Kite","Black Kite","European Honey-buzzard","Western Marsh Harrier","Eurasian Sparrowhawk","Eurasian Kestrel","Osprey","Eurasian Hobby","Hen Harrier","Merlin" logger=csv


# HYPERPARAMETERS OPTIMIZATION
# logs\train\multiruns\2024-09-11_14-58-53
python src/train.py model=unet hparams_search=unet logger=tensorboard
# tensorboard --logdir=logs/train/multiruns
# python src/eval.py model=unet logger=csv

