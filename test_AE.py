import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
from IPython import embed
import lpips

import options as option
from models import create_model

import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

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

os.system("rm ./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

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
lpips_fn = lpips.LPIPS(net='alex').to(device)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []
    test_results["lpips"] = []
    test_times = []

    for i, test_data in enumerate(test_loader):
        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []
        
        need_GT = False if test_loader.dataset.opt["dataroot_X_GT"] is None else True
        img_path = test_data["X_GT_path"][0] if need_GT else test_data["X_LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
 
        

        #### input dataset_LQ
        X_LQ, X_GT = test_data["X_LQ"], test_data["X_GT"]
        Y_LQ, Y_GT = test_data["Y_LQ"], test_data["Y_GT"]
        

        model.feed_data(X_LQ, Y_LQ, X_GT, Y_GT)
        tic = time.time()
        model.test()
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        Get_X_fake_gt = visuals["X_fake_gt"]
        Get_X_fake_lq = visuals["X_fake_lq"]
        Get_Y_fake_gt = visuals["Y_fake_gt"]
        Get_Y_fake_lq = visuals["Y_fake_lq"]
        Get_X_fake_lq_by_Ylq = visuals["X_fake_lq_by_Ylq"]
        Get_X_fake_lq_by_Ygt = visuals["X_fake_lq_by_Ygt"]
        Get_X_fake_gt_by_Ylq = visuals["X_fake_gt_by_Ylq"]
        Get_X_fake_gt_by_Ygt = visuals["X_fake_gt_by_Ygt"]
        Get_Y_fake_lq_by_Xlq = visuals["Y_fake_lq_by_Xlq"]
        Get_Y_fake_lq_by_Xgt = visuals["Y_fake_lq_by_Xgt"]
        Get_Y_fake_gt_by_Xlq = visuals["Y_fake_gt_by_Xlq"]
        Get_Y_fake_gt_by_Xgt = visuals["Y_fake_gt_by_Xgt"]        
     
                        
        output_X_fake_gt = util.tensor2img(Get_X_fake_gt.squeeze())  # uint8
        output_X_fake_lq = util.tensor2img(Get_X_fake_lq.squeeze())  # uint8        
        output_Y_fake_gt = util.tensor2img(Get_Y_fake_gt.squeeze())  # uint8        
        output_Y_fake_lq = util.tensor2img(Get_Y_fake_lq.squeeze())  # uint8        
        output_X_fake_lq_by_Ylq = util.tensor2img(Get_X_fake_lq_by_Ylq.squeeze())  # uint8
        output_X_fake_lq_by_Ygt = util.tensor2img(Get_X_fake_lq_by_Ygt.squeeze())  # uint8
        output_X_fake_gt_by_Ylq = util.tensor2img(Get_X_fake_gt_by_Ylq.squeeze())  # uint8
        output_X_fake_gt_by_Ygt = util.tensor2img(Get_X_fake_gt_by_Ygt.squeeze())  # uint8
        output_Y_fake_lq_by_Xlq = util.tensor2img(Get_Y_fake_lq_by_Xlq.squeeze())  # uint8
        output_Y_fake_lq_by_Xgt = util.tensor2img(Get_Y_fake_lq_by_Xgt.squeeze())  # uint8
        output_Y_fake_gt_by_Xlq = util.tensor2img(Get_Y_fake_gt_by_Xlq.squeeze())  # uint8
        output_Y_fake_gt_by_Xgt = util.tensor2img(Get_Y_fake_gt_by_Xgt.squeeze())  # uint8


        X_LQ_ = util.tensor2img(visuals["X_LQ"].squeeze())  # uint8
        X_GT_ = util.tensor2img(visuals["X_GT"].squeeze())  # uint8
        Y_LQ_ = util.tensor2img(visuals["Y_LQ"].squeeze())  # uint8
        Y_GT_ = util.tensor2img(visuals["Y_GT"].squeeze())  # uint8

        
        suffix = opt["suffix"]
        if suffix:
            save_img_path_X_fake_gt = os.path.join(dataset_dir, img_name + suffix + "_X_fake_gt.png")
            save_img_path_X_fake_lq = os.path.join(dataset_dir, img_name + suffix + "_X_fake_lq.png")
            save_img_path_Y_fake_gt = os.path.join(dataset_dir, img_name + suffix + "_Y_fake_gt.png")
            save_img_path_Y_fake_lq = os.path.join(dataset_dir, img_name + suffix + "_Y_fake_lq.png")
            save_img_path_X_fake_lq_by_Ylq = os.path.join(dataset_dir, img_name + suffix + "_X_fake_lq_by_Ylq.png")
            save_img_path_X_fake_lq_by_Ygt = os.path.join(dataset_dir, img_name + suffix + "_X_fake_lq_by_Ygt.png")
            save_img_path_X_fake_gt_by_Ylq = os.path.join(dataset_dir, img_name + suffix + "_X_fake_gt_by_Ylq.png")
            save_img_path_X_fake_gt_by_Ygt = os.path.join(dataset_dir, img_name + suffix + "_X_fake_gt_by_Ygt.png")
            save_img_path_Y_fake_lq_by_Xlq = os.path.join(dataset_dir, img_name + suffix + "_Y_fake_lq_by_Xlq.png")
            save_img_path_Y_fake_lq_by_Xgt = os.path.join(dataset_dir, img_name + suffix + "_Y_fake_lq_by_Xgt.png")
            save_img_path_Y_fake_gt_by_Xlq = os.path.join(dataset_dir, img_name + suffix + "_Y_fake_gt_by_Xlq.png")
            save_img_path_Y_fake_gt_by_Xgt = os.path.join(dataset_dir, img_name + suffix + "_Y_fake_gt_by_Xgt.png")

        else:
            save_img_path_X_fake_gt = os.path.join(dataset_dir, img_name + "_X_fake_gt.png")
            save_img_path_X_fake_lq = os.path.join(dataset_dir, img_name + "_X_fake_lq.png")
            save_img_path_Y_fake_gt = os.path.join(dataset_dir, img_name + "_Y_fake_gt.png")
            save_img_path_Y_fake_lq = os.path.join(dataset_dir, img_name + "_Y_fake_lq.png")
            save_img_path_X_fake_lq_by_Ylq = os.path.join(dataset_dir, img_name + "_X_fake_lq_by_Ylq.png")
            save_img_path_X_fake_lq_by_Ygt = os.path.join(dataset_dir, img_name + "_X_fake_lq_by_Ygt.png")
            save_img_path_X_fake_gt_by_Ylq = os.path.join(dataset_dir, img_name + "_X_fake_gt_by_Ylq.png")
            save_img_path_X_fake_gt_by_Ygt = os.path.join(dataset_dir, img_name + "_X_fake_gt_by_Ygt.png")
            save_img_path_Y_fake_lq_by_Xlq = os.path.join(dataset_dir, img_name + "_Y_fake_lq_by_Xlq.png")
            save_img_path_Y_fake_lq_by_Xgt = os.path.join(dataset_dir, img_name + "_Y_fake_lq_by_Xgt.png")
            save_img_path_Y_fake_gt_by_Xlq = os.path.join(dataset_dir, img_name + "_Y_fake_gt_by_Xlq.png")
            save_img_path_Y_fake_gt_by_Xgt = os.path.join(dataset_dir, img_name + "_Y_fake_gt_by_Xgt.png")



            
        util.save_img(output_X_fake_gt, save_img_path_X_fake_gt)
        util.save_img(output_X_fake_lq, save_img_path_X_fake_lq)
        util.save_img(output_Y_fake_gt, save_img_path_Y_fake_gt)
        util.save_img(output_Y_fake_lq, save_img_path_Y_fake_lq)
        util.save_img(output_X_fake_lq_by_Ylq, save_img_path_X_fake_lq_by_Ylq)
        util.save_img(output_X_fake_lq_by_Ygt, save_img_path_X_fake_lq_by_Ygt)
        util.save_img(output_X_fake_gt_by_Ylq, save_img_path_X_fake_gt_by_Ylq)
        util.save_img(output_X_fake_gt_by_Ygt, save_img_path_X_fake_gt_by_Ygt)
        util.save_img(output_Y_fake_lq_by_Xlq, save_img_path_Y_fake_lq_by_Xlq)
        util.save_img(output_Y_fake_lq_by_Xgt, save_img_path_Y_fake_lq_by_Xgt)
        util.save_img(output_Y_fake_gt_by_Xlq, save_img_path_Y_fake_gt_by_Xlq)
        util.save_img(output_Y_fake_gt_by_Xgt, save_img_path_Y_fake_gt_by_Xgt)

                

        # remove it if you only want to save output images
        X_LQ_img_path = os.path.join(dataset_dir, img_name + "_X_LQ.png")
        X_GT_img_path = os.path.join(dataset_dir, img_name + "_X_HQ.png")
        Y_LQ_img_path = os.path.join(dataset_dir, img_name + "_Y_LQ.png")
        Y_GT_img_path = os.path.join(dataset_dir, img_name + "_Y_HQ.png")
        util.save_img(X_LQ_, X_LQ_img_path)
        util.save_img(X_GT_, X_GT_img_path)
        util.save_img(Y_LQ_, Y_LQ_img_path)
        util.save_img(Y_GT_, Y_GT_img_path)


        if need_GT:
            gt_img = X_GT_ / 255.0
            sr_img = output_X_fake_gt_by_Ylq / 255.0

            crop_border = opt["crop_border"] if opt["crop_border"] else 0
            if crop_border == 0:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
            else:
                cropped_sr_img = sr_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]
                cropped_gt_img = gt_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]

            psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)


            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)
          

            if len(gt_img.shape) == 3:
                if gt_img.shape[2] == 3: 
                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    if crop_border == 0:
                        cropped_sr_img_y = sr_img_y
                        cropped_gt_img_y = gt_img_y
                    else:
                        cropped_sr_img_y = sr_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                        cropped_gt_img_y = gt_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                    psnr_y = util.calculate_psnr(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )
                    ssim_y = util.calculate_ssim(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )

                    test_results["psnr_y"].append(psnr_y)
                    test_results["ssim_y"].append(ssim_y)



                    logger.info(
                        "img{:3d}:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f};  PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.".format(
                            i, img_name, psnr, ssim, psnr_y, ssim_y
                        )
                    )

            else:
                logger.info(
                    "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}.".format(
                        img_name, psnr, ssim
                    )
                )

                test_results["psnr_y"].append(psnr)
                test_results["ssim_y"].append(ssim)
        else:
            logger.info(img_name)


    ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
    ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
    logger.info(
        "----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
            test_set_name, ave_psnr, ave_ssim
        )
    )
    if test_results["psnr_y"] and test_results["ssim_y"]:
        ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
        ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
        logger.info(
            "----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n".format(
                ave_psnr_y, ave_ssim_y
            )
        )


    print(f"average test time: {np.mean(test_times):.4f}")