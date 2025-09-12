from typing import Any, Dict, List, Tuple
import time 

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
OmegaConf.register_new_resolver("len", len)
OmegaConf.register_new_resolver("eval", eval)

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for prediction.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    assert cfg.ckpt_path_pred

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=False)

    log.info("Starting predictions!")
    trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path_pred)
    # https://www.trektellen.org/species/graph/3/2422/101/0?g=&l=&k=&jaar2=&jaar3=&graphtype1=bar&graphtype2=line&graphtype3=line&hidempbars=1&
    # https://www.trektellen.org/count/view/2422/20240919
    log.info("Predicting finished!")


if __name__ == "__main__":
    main()
