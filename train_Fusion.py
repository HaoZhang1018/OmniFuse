import argparse
import logging
import math
import os
import shutil
import random
import sys
import copy
import torchvision.utils as tvutils
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# from IPython import embed
from PIL import Image
import options as option
from models import create_model
import gc
sys.path.insert(0, "./models")
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr

# torch.autograd.set_detect_anomaly(True)

def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")


    #### create train and val dataloader
    dataset_ratio = 100  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            test_set_name = val_loader.dataset.opt["name"]
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None
    print("create dataset done!!")
    #### create model
    model = create_model(opt) 
    device = model.device
    print("create model done!!")
    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    sde_x = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde_x.set_model(model.diff_model_X)

    sde_y = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde_y.set_model(model.diff_model_Y)

    scale = opt['degradation']['scale']

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_cos_sim = np.inf
    best_iter = 0
    error = mp.Value('b', False)
    print("start to training...")
    #idex=0
    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1

            if current_step > total_iters:
                break
# #####################################################################################
            if opt['Fusion_Model_type']=='base':
            # get input feature
                Modal_X, Modal_Y = train_data["Modal_X"], train_data["Modal_Y"]
                Modal_X = Modal_X.to(device)
                Modal_Y = Modal_Y.to(device)    
                latent_X, hidden_X = model.encode(Modal_X)
                latent_Y, hidden_Y = model.encode(Modal_Y)

                noisy_state_X = sde_x.noise_state(latent_X)
                sde_x.set_mu(latent_X)
                Modal_latent_X = sde_x.reverse_sde(noisy_state_X, save_states=False)
                        
                noisy_state_Y = sde_y.noise_state(latent_Y)
                sde_y.set_mu(latent_Y)
                Modal_latent_Y = sde_y.reverse_sde(noisy_state_Y, save_states=False)
            
                Modal_X_fea = [Modal_latent_X] + hidden_X
                Modal_Y_fea = [Modal_latent_Y] + hidden_Y 
                model.feed_data(Modal_X_fea, Modal_Y_fea)
            
                model.optimize_parameters(current_step)
                model.update_learning_rate(
                    current_step, warmup_iter=opt["train"]["warmup_iter"]
                )

                if current_step % opt["logger"]["print_freq"] == 0:
                    logs = model.get_current_log()
                    message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                        epoch, current_step, model.get_current_learning_rate()
                    )
                    for k, v in logs.items():
                        message += "{:s}: {:.4e} ".format(k, v)
                        # tensorboard logger
                        if opt["use_tb_logger"] and "debug" not in opt["name"]:
                            if rank <= 0:
                                tb_logger.add_scalar(k, v, current_step)
                    if rank <= 0:
                        logger.info(message)

            # validation, to produce ker_map_list(fake)
                if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                    avg_cos_sim = 0.0
                    idx = 0
                    for _, val_data in enumerate(val_loader):
                         if 1>0:#torch.randint(0, 11, (1,)).item()==5:
                             need_GT = False if val_loader.dataset.opt["dataroot_GT"] is None else True
                             img_path = val_data["Modal_X_path"][0] if need_GT else val_data["Modal_Y_path"][0]
                             img_name = os.path.splitext(os.path.basename(img_path))[0]
                             Modal_X, Modal_Y = val_data["Modal_X"], val_data["Modal_Y"]
                             Modal_X = Modal_X.to(device)
                             Modal_Y = Modal_Y.to(device)
                
                             with torch.no_grad():
                                 latent_X, hidden_X = model.encode(Modal_X)
                                 latent_Y, hidden_Y = model.encode(Modal_Y)
                        
                                 noisy_state_X = sde_x.noise_state(latent_X)
                                 sde_x.set_mu(latent_X)                        
                                 Modal_latent_X = sde_x.reverse_sde(noisy_state_X, save_states=False) 
                        
                                 noisy_state_Y = sde_y.noise_state(latent_Y)
                                 sde_y.set_mu(latent_Y)
                                 Modal_latent_Y = sde_y.reverse_sde(noisy_state_Y, save_states=False)
                            # Modal_latent_X = latent_X
                            # Modal_latent_Y = latent_Y

                                 Modal_X_fea = [Modal_latent_X] + hidden_X
                                 Modal_Y_fea = [Modal_latent_Y] + hidden_Y

                    # valid Predictor
                             model.feed_data(Modal_X_fea, Modal_Y_fea)
                             model.test()
                             visuals = model.get_current_visuals()

                             idx += 1
        
                             output = visuals["Output"].squeeze()  # uint8
                             output_imgname = f'image/{img_name}.png'
                             tvutils.save_image(output, output_imgname, normalize=False)

                    # break
                             gc.collect()
                             torch.cuda.empty_cache()
        
            if opt['Fusion_Model_type']=='modulated':
            # get input feature
                Modal_X, Modal_Y,Seg_Label,Fusion_Base = train_data["Modal_X"], train_data["Modal_Y"],train_data["Seg_Label"], train_data["Fusion_Base"]
                Modal_X = Modal_X.to(device)
                Modal_Y = Modal_Y.to(device)    
                Seg_Label = Seg_Label.to(device)
                Fusion_Base = Fusion_Base.to(device)    
                latent_X, hidden_X = model.encode(Modal_X)
                latent_Y, hidden_Y = model.encode(Modal_Y)

                noisy_state_X = sde_x.noise_state(latent_X)
                sde_x.set_mu(latent_X)
                Modal_latent_X = sde_x.reverse_sde(noisy_state_X, save_states=False)
                        
                noisy_state_Y = sde_y.noise_state(latent_Y)
                sde_y.set_mu(latent_Y)
                Modal_latent_Y = sde_y.reverse_sde(noisy_state_Y, save_states=False)
            
                Modal_X_fea = [Modal_latent_X] + hidden_X
                Modal_Y_fea = [Modal_latent_Y] + hidden_Y 
                prompt = train_data["prompt"]
                model.add_dataset(Seg_Label, Fusion_Base)
                model.feed_data(Modal_X_fea, Modal_Y_fea,Fusion_Model_type=opt["Fusion_Model_type"], prompt=prompt)
            
                model.optimize_parameters(current_step,Fusion_Model_type=opt["Fusion_Model_type"])
                model.update_learning_rate(
                    current_step, warmup_iter=opt["train"]["warmup_iter"]
                )

                if current_step % opt["logger"]["print_freq"] == 0:
                    logs = model.get_current_log()
                    message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                        epoch, current_step, model.get_current_learning_rate()
                    )
                    for k, v in logs.items():
                        message += "{:s}: {:.4e} ".format(k, v)
                        # tensorboard logger
                        if opt["use_tb_logger"] and "debug" not in opt["name"]:
                            if rank <= 0:
                                tb_logger.add_scalar(k, v, current_step)
                    if rank <= 0:
                        logger.info(message)

            # validation, to produce ker_map_list(fake)
                if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                    avg_cos_sim = 0.0
                    idx = 0
                    for _, val_data in enumerate(val_loader):
                         if 1>0:#torch.randint(0, 11, (1,)).item()==5:
                             need_GT = False if val_loader.dataset.opt["dataroot_GT"] is None else True
                             img_path = val_data["Modal_X_path"][0] if need_GT else val_data["Modal_Y_path"][0]
                             img_name = os.path.splitext(os.path.basename(img_path))[0]
                             Modal_X, Modal_Y = val_data["Modal_X"], val_data["Modal_Y"]
                             Modal_X = Modal_X.to(device)
                             Modal_Y = Modal_Y.to(device)
                
                             with torch.no_grad():
                                 latent_X, hidden_X = model.encode(Modal_X)
                                 latent_Y, hidden_Y = model.encode(Modal_Y)
                        
                                 noisy_state_X = sde_x.noise_state(latent_X)
                                 sde_x.set_mu(latent_X)                        
                                 Modal_latent_X = sde_x.reverse_sde(noisy_state_X, save_states=False) 
                        
                                 noisy_state_Y = sde_y.noise_state(latent_Y)
                                 sde_y.set_mu(latent_Y)
                                 Modal_latent_Y = sde_y.reverse_sde(noisy_state_Y, save_states=False)


                                 Modal_X_fea = [Modal_latent_X] + hidden_X
                                 Modal_Y_fea = [Modal_latent_Y] + hidden_Y

                    # valid Predictor
                             model.feed_data(Modal_X_fea, Modal_Y_fea)
                             model.test()
                             visuals = model.get_current_visuals()

                             idx += 1
                             output = visuals["Output"].squeeze()  # uint8
                             output_imgname = f'image/{img_name}.png'
                             tvutils.save_image(output, output_imgname, normalize=False)

                    
                    # break
                             gc.collect()
                             torch.cuda.empty_cache()  
              
            if error.value:
                sys.exit(0)
            #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
        tb_logger.close()

if __name__ == "__main__":
    main()