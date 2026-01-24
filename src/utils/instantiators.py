from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, (DictConfig, ListConfig, list)):
        raise TypeError("Callbacks config must be a DictConfig or ListConfig!")

    def _maybe_instantiate(cb_conf) -> None:
        if isinstance(cb_conf, Callback):
            callbacks.append(cb_conf)
            return

        if isinstance(cb_conf, DictConfig):
            # Standard hydra config with _target_
            target = cb_conf.get("_target_", None)
            if target:
                log.info(f"Instantiating callback <{target}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
                return

            # Structured configs (object_type set) -> convert to object
            obj_type = OmegaConf.get_type(cb_conf)
            if obj_type is not None:
                log.info(f"Instantiating callback <{obj_type}>")
                callbacks.append(OmegaConf.to_object(cb_conf))
                return

        # Ignore unsupported entries

    if isinstance(callbacks_cfg, DictConfig):
        for _, cb_conf in callbacks_cfg.items():
            _maybe_instantiate(cb_conf)
    else:
        for cb_conf in list(callbacks_cfg):
            _maybe_instantiate(cb_conf)

    # Final safety pass: convert any remaining DictConfig entries
    final_callbacks: List[Callback] = []
    for cb in callbacks:
        if isinstance(cb, DictConfig):
            target = cb.get("_target_", None)
            if target:
                log.info(f"Instantiating callback <{target}>")
                cb = hydra.utils.instantiate(cb)
            else:
                obj_type = OmegaConf.get_type(cb)
                if obj_type is not None:
                    log.info(f"Instantiating callback <{obj_type}>")
                    cb = OmegaConf.to_object(cb)
        if not isinstance(cb, Callback):
            raise TypeError(f"Callback is not instantiated properly: {type(cb)}")
        final_callbacks.append(cb)

    return final_callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
