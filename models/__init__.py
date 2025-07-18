import logging

logger = logging.getLogger("base")


def create_model(opt):
    model = opt["model"]


    if model == "sde":
        from .sde_model import SDEModel as M
    elif model == 'latent':
        from .latent_model import LatentModel as M
    elif model == "Fusion":
        from .fusion_model import FusionModel as M
    elif model == "latent_denoising":
        from .latent_denoising_model import DenoisingModel as M
    else:
        raise NotImplementedError("Model [{:s}] not recognized.".format(model))
    m = M(opt)
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m
