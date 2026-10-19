# BatchNorm ordering in the daily branch

The daily network's blocks are Conv → BatchNorm → ReLU. With the earlier Conv → ReLU → BatchNorm,
a channel whose pre-activation is negative on almost every day has a running variance near zero
(down to 1e-11 measured), because ReLU removes the variation before BatchNorm sees it. In eval
mode BatchNorm then scales that channel by up to about 400x, so on the rare day it fires the logit
reaches hundreds and the daily output saturates, predicting a large count for an absent species.
Training mode uses batch statistics, so the loss never showed it. Putting BatchNorm before the
ReLU keeps the variance. `tests/test_unet.py` checks this mechanism directly, without training.

Checkpoints trained with the old order do not load into the new layers, so the change landed
together with a retrain of all species (`DECISIONS.md` → Retrain on the release counts). The effect
of the order on forecast skill, apart from the data and prior changes, was not measured.
