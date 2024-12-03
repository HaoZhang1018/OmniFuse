import logging

import torch

from models import modules as M

logger = logging.getLogger("base")

# Generator
def define_Diff(opt):
    opt_net = opt["network_Diff"]
    which_model = opt_net["which_model"]
    setting = opt_net["setting"]
    netDiff = getattr(M, which_model)(**setting)
    return netDiff


# Latent model
def define_AE(opt):
    opt_net = opt["network_AE"]
    which_model = opt_net["which_model"]
    setting = opt_net["setting"]
    netAE = getattr(M, which_model)(**setting)
    return netAE


# Latent Fuse model
def define_Fuse(opt):
    opt_net = opt["network_Fuse"]
    which_model = opt_net["which_model"]
    setting = opt_net["setting"]
    netFuse = getattr(M, which_model)(**setting)
    return netFuse

