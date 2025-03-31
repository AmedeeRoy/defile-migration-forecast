#!/bin/bash

# BASELINE HYPERPARAMETERS OPTIMIZATION
# To see the results:
# cd logs/mlflow
# mlflow ui --port=3000
nohup python src/train.py task_name=optim data.species="Common Buzzard" hparams_search=unet trainer=gpu logger=mlflow &
nohup python src/train.py task_name=optim data.species="Red Kite" hparams_search=unet trainer=gpu logger=mlflow &
nohup python src/train.py task_name=optim data.species="Black Kite" hparams_search=unet trainer=gpu logger=mlflow &
nohup python src/train.py task_name=optim data.species="European Honey-buzzard" hparams_search=unet trainer=gpu logger=mlflow &
nohup python src/train.py task_name=optim data.species="Western Marsh Harrier" hparams_search=unet trainer=gpu logger=mlflow &
nohup python src/train.py task_name=optim data.species="Eurasian Sparrowhawk" hparams_search=unet trainer=gpu logger=mlflow &
nohup python src/train.py task_name=optim data.species="Eurasian Kestrel" hparams_search=unet trainer=gpu logger=mlflow &
nohup python src/train.py task_name=optim data.species="Osprey" hparams_search=unet trainer=gpu logger=mlflow &
nohup python src/train.py task_name=optim data.species="Hen Harrier" hparams_search=unet trainer=gpu logger=mlflow &
nohup python src/train.py task_name=optim data.species="Merlin" hparams_search=unet trainer=gpu logger=mlflow &
nohup python src/train.py task_name=optim data.species="Eurasian Hobby" hparams_search=unet trainer=gpu logger=mlflow &

# DEFINE SPECIES-SPEFICIC EXPERIMENTS
python src/train.py --multirun experiment=common_buzzard,red_kite,black_kite,honey_buzzard,marsh_harrier,sparrowhawk,kestrel,osprey,hen_harrier,merlin,hobby trainer=gpu

# # PREDICT
# python src/predict.py data.species="Common Buzzard"

# # PRODUCTION
# python src/train.py --multirun experiment=production data.species="Common Buzzard","Red Kite","Black Kite","European Honey-buzzard","Western Marsh Harrier","Eurasian Sparrowhawk","Eurasian Kestrel","Osprey","Eurasian Hobby","Hen Harrier","Merlin" logger=csv
# python src/predict.py --multirun experiment=production data.species="Common Buzzard","Red Kite","Black Kite","European Honey-buzzard","Western Marsh Harrier","Eurasian Sparrowhawk","Eurasian Kestrel","Osprey","Eurasian Hobby","Hen Harrier","Merlin" logger=csv
python src/predict.py data.species="Common Buzzard"