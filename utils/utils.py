import os

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.functional import normalize
import typing as t
from utils.logger import logger


def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean / std
    _std = 1 / std
    return normalize(tensor, _mean, _std)


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean / std
        self._std = 1 / std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1, 1, 1)) / self._std.reshape(-1, 1, 1)
        return normalize(tensor, self._mean, self._std)


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def save_ckpt(cur_itrs, path, *, model, optimizer, scheduler, best_score, scaler):
    """ save current model
    """
    torch.save({
        "cur_itrs": cur_itrs,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
        "scaler": scaler.state_dict(),
    }, path)
    logger.trace("Model saved as %s" % path)


def get_lrs_from_optimizer(optimizer) -> t.List[float]:
    return [group["lr"] for group in optimizer.param_groups]
