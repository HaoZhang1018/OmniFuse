import os
import sys
import time
import logging
import argparse

import numpy as np

from IPython import embed

import options as option
from models import create_model

sys.path.insert(0, "./models")

import utils as util
from data import create_dataloader, create_dataset

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
parser.add_argument("-prompt", type=list, default=["Pay attention to person"], help="Prompt to highlight the target. If empty, not used.")


args = parser.parse_args()

opt = option.parse(args.opt, is_train=False)
opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device

sde_x = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde_x.set_model(model.diff_model_X)

sde_y = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde_y.set_model(model.diff_model_Y)

# start to test

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    
    print(opt["path"]["results_root"])
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    for i, test_data in enumerate(test_loader):
        Initial_H, Initial_W = test_data["Initial_H"], test_data["Initial_W"]
        img_path = test_data["Modal_X_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(img_name)

        #### input dataset_LQ
        Modal_X, Modal_Y = test_data["Modal_X"], test_data["Modal_Y"]
        B, C, H, W = Modal_X.shape
    
        latent_X, hidden_X = model.encode(Modal_X.to(device))
        latent_Y, hidden_Y = model.encode(Modal_Y.to(device))

        noisy_state_X = sde_x.noise_state(latent_X)
        sde_x.set_mu(latent_X)                        
        Modal_latent_X = sde_x.reverse_sde(noisy_state_X, save_states=False) 

        noisy_state_Y = sde_y.noise_state(latent_Y)
        sde_y.set_mu(latent_Y)
        Modal_latent_Y = sde_y.reverse_sde(noisy_state_Y, save_states=False)        
    
        Modal_X_fea = [Modal_latent_X] + hidden_X
        Modal_Y_fea = [Modal_latent_Y] + hidden_Y
        prompt = test_data["prompt"] if opt["datasets"]["test"]["With_prompt"] else args.prompt

        model.add_dataset()
        model.feed_data(Modal_X_fea, Modal_Y_fea, Fusion_Model_type=opt["Fusion_Model_type"], prompt=prompt)
        model.test()

        visuals = model.get_current_visuals()
        Fus_img = visuals["Output"][None, ...]

        
        output = util.tensor2img(Fus_img.squeeze())  # uint8


        suffix = opt["suffix"]
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(dataset_dir, img_name + ".png")

        output = output[:Initial_H, :Initial_W, :]
        util.save_img(output, save_img_path)
