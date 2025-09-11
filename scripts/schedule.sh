#!/bin/bash

# BASELINE HYPERPARAMETERS OPTIMIZATION
# To see the results:
# cd logs/mlflow
# mlflow ui --port=3000
nohup python src/train.py task_name=optim data.species="Common Buzzard" hparams_search=unet trainer=gpu  &
nohup python src/train.py task_name=optim data.species="Red Kite" hparams_search=unet trainer=gpu  &
nohup python src/train.py task_name=optim data.species="Black Kite" hparams_search=unet trainer=gpu  &
nohup python src/train.py task_name=optim data.species="European Honey-buzzard" hparams_search=unet trainer=gpu  &
nohup python src/train.py task_name=optim data.species="Western Marsh Harrier" hparams_search=unet trainer=gpu  &
nohup python src/train.py task_name=optim data.species="Eurasian Sparrowhawk" hparams_search=unet trainer=gpu  &
nohup python src/train.py task_name=optim data.species="Eurasian Kestrel" hparams_search=unet trainer=gpu  &
nohup python src/train.py task_name=optim data.species="Osprey" hparams_search=unet trainer=gpu  &
nohup python src/train.py task_name=optim data.species="Hen Harrier" hparams_search=unet trainer=gpu  &
nohup python src/train.py task_name=optim data.species="Merlin" hparams_search=unet trainer=gpu  &
nohup python src/train.py task_name=optim data.species="Eurasian Hobby" hparams_search=unet trainer=gpu  &

# TRAINING
# All model at once:
python src/train.py --multirun experiment=common_buzzard,red_kite,black_kite,honey_buzzard,marsh_harrier,sparrowhawk,kestrel,osprey,hen_harrier,merlin,hobby trainer=gpu

# # MAKE PROD: Move checkpoints to prod
# python scripts/move_checkpoints_to_prod.py --dry-run --force
# # Only multirun experiments
# python scripts/move_checkpoints_to_prod.py --run-type multiruns
# # Only multirun experiments
# python scripts/move_checkpoints_to_prod.py --run-type multiruns

# # PREDICT
# python src/predict.py data.species="Black Kite"

python src/predict.py experiment=honey_buzzard
python src/predict.py --multirun experiment=common_buzzard,red_kite,black_kite,honey_buzzard,marsh_harrier,sparrowhawk,kestrel,osprey,hen_harrier,merlin,hobby trainer=gpu
python src/predict.py --multirun experiment=black_kite,honey_buzzard,marsh_harrier,sparrowhawk,kestrel,osprey trainer=gpu
