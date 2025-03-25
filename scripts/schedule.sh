#!/bin/bash

# TRAIN
# Main training with all data
python src/train.py

python src/train.py data="defile-small"

python src/train.py debug=default

# PREDICT
python src/predict.py data.species="Common Buzzard"

# PRODUCTION
python src/train.py --multirun experiment=production data.species="Common Buzzard","Red Kite","Black Kite","European Honey-buzzard","Western Marsh Harrier","Eurasian Sparrowhawk","Eurasian Kestrel","Osprey","Eurasian Hobby","Hen Harrier","Merlin" logger=csv
python src/predict.py --multirun experiment=production data.species="Common Buzzard","Red Kite","Black Kite","European Honey-buzzard","Western Marsh Harrier","Eurasian Sparrowhawk","Eurasian Kestrel","Osprey","Eurasian Hobby","Hen Harrier","Merlin" logger=csv


# HYPERPARAMETERS OPTIMIZATION
# logs\train\multiruns\2024-09-11_14-58-53
python src/train.py data.species="Common Buzzard" hparams_search=unet logger=mlflow
python src/train.py data.species="Red Kite" hparams_search=unet logger=mlflow
python src/train.py data.species="Black Kite" hparams_search=unet logger=mlflow
python src/train.py data.species="European Honey-buzzard" hparams_search=unet logger=mlflow
python src/train.py data.species="Eurasian Sparrowhawk" hparams_search=unet logger=mlflow
python src/train.py data.species="Eurasian Kestrel" hparams_search=unet logger=mlflow
python src/train.py data.species="Osprey" hparams_search=unet logger=mlflow
python src/train.py data.species="Hen Harrier" hparams_search=unet logger=mlflow
python src/train.py data.species="Merlin" hparams_search=unet logger=mlflow


# tensorboard --logdir=logs/train/multiruns
# python src/eval.py model=unet logger=csv

