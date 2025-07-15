import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion
from models.modules.loss import MatchingLoss, PerceptualMatchingLoss

from .base_model import BaseModel

logger = logging.getLogger("base")


class LatentModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.model = networks.define_AE(opt).to(self.device)
        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )
        # else:
        #     self.model = DataParallel(self.model)
        # print network
        # self.print_network()
        self.load()

        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            # self.loss_fn = PerceptualMatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt['train']['weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "CosineAnnealingLR_Restart":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer,
                            train_opt["T_period"],
                            eta_min=train_opt["eta_min"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            # self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()

    def feed_data(self, X_LQ, Y_LQ, X_GT=None, Y_GT=None):
        self.X_LQ = X_LQ.to(self.device)  # LQ
        self.X_GT = X_GT.to(self.device) if X_GT is not None else None
        self.Y_LQ = Y_LQ.to(self.device)  # LQ
        self.Y_GT = Y_GT.to(self.device) if Y_GT is not None else None


    def optimize_parameters(self, step):
        self.optimizer.zero_grad()

        if self.opt["dist"]:
            encode_fn = self.model.module.encode
            decode_fn = self.model.module.decode
        else:
            encode_fn = self.model.encode
            decode_fn = self.model.decode

        X_L_lq, X_H_lq = encode_fn(self.X_LQ)
        X_L_gt, X_H_gt = encode_fn(self.X_GT)
        Y_L_lq, Y_H_lq = encode_fn(self.Y_LQ)
        Y_L_gt, Y_H_gt = encode_fn(self.Y_GT)

        
        X_rec_llq_hlq = decode_fn(X_L_lq, X_H_lq) ## obtain X true LQ 
        X_rec_llq_hgt = decode_fn(X_L_lq, X_H_gt) ## obtain X fake LQ
        
        X_rec_lgt_hgt = decode_fn(X_L_gt, X_H_gt) ## obtain X true GT
        X_rec_lgt_hlq = decode_fn(X_L_gt, X_H_lq) ## obtain X fake GT
        
        Y_rec_llq_hlq = decode_fn(Y_L_lq, Y_H_lq) ## obtain Y true LQ
        Y_rec_llq_hgt = decode_fn(Y_L_lq, Y_H_gt) ## obtain Y fake LQ
        
        Y_rec_lgt_hgt = decode_fn(Y_L_gt, Y_H_gt) ## obtain Y true GT
        Y_rec_lgt_hlq = decode_fn(Y_L_gt, Y_H_lq) ## obtain Y fake GT        
        
        
        rec_X_llq_Y_hlq = decode_fn(X_L_lq, Y_H_lq) ## obtain X fake LQ
        rec_X_llq_Y_hgt = decode_fn(X_L_lq, Y_H_gt) ## obtain X fake LQ
        
        rec_X_lgt_Y_hlq = decode_fn(X_L_gt, Y_H_lq) ## obtain X fake GT
        rec_X_lgt_Y_hgt = decode_fn(X_L_gt, Y_H_gt) ## obtain X fake GT


        rec_Y_llq_X_hlq = decode_fn(Y_L_lq, X_H_lq) ## obtain Y fake LQ
        rec_Y_llq_X_hgt = decode_fn(Y_L_lq, X_H_gt) ## obtain Y fake LQ
        
        rec_Y_lgt_X_hlq = decode_fn(Y_L_gt, X_H_lq) ## obtain Y fake GT
        rec_Y_lgt_X_hgt = decode_fn(Y_L_gt, X_H_gt) ## obtain Y fake GT


        

        X_loss_rec = self.loss_fn(X_rec_llq_hlq, self.X_LQ) + self.loss_fn(X_rec_lgt_hgt, self.X_GT) + self.loss_fn(X_rec_lgt_hlq, self.X_GT) + self.loss_fn(X_rec_llq_hgt, self.X_LQ)
        Y_loss_rec = self.loss_fn(Y_rec_llq_hlq, self.Y_LQ) + self.loss_fn(Y_rec_lgt_hgt, self.Y_GT) + self.loss_fn(Y_rec_lgt_hlq, self.Y_GT) + self.loss_fn(Y_rec_llq_hgt, self.Y_LQ)
        X_Y_loss_rec = self.loss_fn(rec_X_llq_Y_hlq, self.X_LQ) + self.loss_fn(rec_X_lgt_Y_hlq, self.X_GT) + self.loss_fn(rec_X_lgt_Y_hgt, self.X_GT) + self.loss_fn(rec_X_llq_Y_hgt, self.X_LQ)
        Y_X_loss_rec = self.loss_fn(rec_Y_llq_X_hlq, self.Y_LQ) + self.loss_fn(rec_Y_lgt_X_hlq, self.Y_GT) + self.loss_fn(rec_Y_lgt_X_hgt, self.Y_GT) + self.loss_fn(rec_Y_llq_X_hgt, self.Y_LQ)

       
        
        

        loss = X_loss_rec + Y_loss_rec + X_Y_loss_rec + Y_X_loss_rec# + loss_reg * 0.001
        loss.backward()
        self.optimizer.step()

        # set log
        self.log_dict["X_loss_rec"] = X_loss_rec.item()
        self.log_dict["Y_loss_rec"] = Y_loss_rec.item()
        self.log_dict["X_Y_loss_rec"] = X_Y_loss_rec.item()
        self.log_dict["Y_X_loss_rec"] = Y_X_loss_rec.item()

    def test(self):
        self.model.eval()

        if self.opt["dist"]:
            encode_fn = self.model.module.encode
            decode_fn = self.model.module.decode
        else:
            encode_fn = self.model.encode
            decode_fn = self.model.decode

        with torch.no_grad():
            X_L_lq, X_H_lq = encode_fn(self.X_LQ)
            X_L_gt, X_H_gt = encode_fn(self.X_GT)
            Y_L_lq, Y_H_lq = encode_fn(self.Y_LQ)
            Y_L_gt, Y_H_gt = encode_fn(self.Y_GT)


            self.X_real_lq = decode_fn(X_L_lq, X_H_lq) # latent LQ, hidden LQ
            self.X_fake_gt = decode_fn(X_L_gt, X_H_lq) # latent GT, hidden LQ

            self.X_fake_lq = decode_fn(X_L_lq, X_H_gt) # latent LQ, hidden GT
            self.X_real_gt = decode_fn(X_L_gt, X_H_gt) # latent GT, hidden GT

            self.Y_real_lq = decode_fn(Y_L_lq, Y_H_lq) # latent LQ, hidden LQ
            self.Y_fake_gt = decode_fn(Y_L_gt, Y_H_lq) # latent GT, hidden LQ

            self.Y_fake_lq = decode_fn(Y_L_lq, Y_H_gt) # latent LQ, hidden GT
            self.Y_real_gt = decode_fn(Y_L_gt, Y_H_gt) # latent GT, hidden GT            


            self.X_fake_lq_by_Ylq = decode_fn(X_L_lq, Y_H_lq) # latent LQ, hidden LQ
            self.X_fake_lq_by_Ygt = decode_fn(X_L_lq, Y_H_gt) # latent LQ, hidden LQ


            self.X_fake_gt_by_Ylq = decode_fn(X_L_gt, Y_H_lq) # latent LQ, hidden LQ
            self.X_fake_gt_by_Ygt = decode_fn(X_L_gt, Y_H_gt) # latent LQ, hidden LQ            


            self.Y_fake_lq_by_Xlq = decode_fn(Y_L_lq, X_H_lq) # latent LQ, hidden LQ
            self.Y_fake_lq_by_Xgt = decode_fn(Y_L_lq, X_H_gt) # latent LQ, hidden LQ


            self.Y_fake_gt_by_Xlq = decode_fn(Y_L_gt, X_H_lq) # latent LQ, hidden LQ
            self.Y_fake_gt_by_Xgt = decode_fn(Y_L_gt, X_H_gt) # latent LQ, hidden LQ   
            

        self.model.train()
        test_folder = './image/'
        if not os.path.exists(test_folder): 
          os.makedirs(test_folder)

        tvutils.save_image(self.X_LQ.data, f'image/X_LQ.png', normalize=False)
        tvutils.save_image(self.X_GT.data, f'image/X_GT.png', normalize=False)

        tvutils.save_image(self.Y_LQ.data, f'image/Y_LQ.png', normalize=False)
        tvutils.save_image(self.Y_GT.data, f'image/Y_GT.png', normalize=False)
        
        tvutils.save_image(self.X_fake_gt.data, f'image/X_GT_fake.png', normalize=False)
        tvutils.save_image(self.X_fake_lq.data, f'image/X_LQ_fake.png', normalize=False)
        tvutils.save_image(self.X_real_gt.data, f'image/X_GT_real.png', normalize=False)
        tvutils.save_image(self.X_real_lq.data, f'image/X_LQ_real.png', normalize=False)

        tvutils.save_image(self.Y_fake_gt.data, f'image/Y_GT_fake.png', normalize=False)
        tvutils.save_image(self.Y_fake_lq.data, f'image/Y_LQ_fake.png', normalize=False)
        tvutils.save_image(self.Y_real_gt.data, f'image/Y_GT_real.png', normalize=False)
        tvutils.save_image(self.Y_real_lq.data, f'image/Y_LQ_real.png', normalize=False)

        tvutils.save_image(self.X_fake_lq_by_Ylq.data, f'image/X_LQ_fake_by_Ylq.png', normalize=False)
        tvutils.save_image(self.X_fake_lq_by_Ygt.data, f'image/X_LQ_fake_by_Ygt.png', normalize=False)
        tvutils.save_image(self.X_fake_gt_by_Ylq.data, f'image/X_GT_fake_by_Ylq.png', normalize=False)
        tvutils.save_image(self.X_fake_gt_by_Ygt.data, f'image/X_GT_fake_by_Ygt.png', normalize=False)


        tvutils.save_image(self.Y_fake_lq_by_Xlq.data, f'image/Y_LQ_fake_by_Xlq.png', normalize=False)
        tvutils.save_image(self.Y_fake_lq_by_Xgt.data, f'image/Y_LQ_fake_by_Xgt.png', normalize=False)
        tvutils.save_image(self.Y_fake_gt_by_Xlq.data, f'image/Y_GT_fake_by_Xlq.png', normalize=False)
        tvutils.save_image(self.Y_fake_gt_by_Xgt.data, f'image/Y_GT_fake_by_Xgt.png', normalize=False)


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["X_LQ"] = self.X_LQ.detach()[0].float().cpu()
        out_dict["Y_LQ"] = self.Y_LQ.detach()[0].float().cpu()
        out_dict["X_fake_gt"] = self.X_fake_gt.detach()[0].float().cpu()
        out_dict["X_fake_lq"] = self.X_fake_lq.detach()[0].float().cpu()        
        out_dict["Y_fake_gt"] = self.Y_fake_gt.detach()[0].float().cpu()
        out_dict["Y_fake_lq"] = self.Y_fake_lq.detach()[0].float().cpu()         
        out_dict["X_fake_lq_by_Ylq"] = self.X_fake_lq_by_Ylq.detach()[0].float().cpu()
        out_dict["X_fake_lq_by_Ygt"] = self.X_fake_lq_by_Ygt.detach()[0].float().cpu()   
        out_dict["X_fake_gt_by_Ylq"] = self.X_fake_gt_by_Ylq.detach()[0].float().cpu()
        out_dict["X_fake_gt_by_Ygt"] = self.X_fake_gt_by_Ygt.detach()[0].float().cpu()   
        out_dict["Y_fake_lq_by_Xlq"] = self.Y_fake_lq_by_Xlq.detach()[0].float().cpu()
        out_dict["Y_fake_lq_by_Xgt"] = self.Y_fake_lq_by_Xgt.detach()[0].float().cpu()   
        out_dict["Y_fake_gt_by_Xlq"] = self.Y_fake_gt_by_Xlq.detach()[0].float().cpu()
        out_dict["Y_fake_gt_by_Xgt"] = self.Y_fake_gt_by_Xgt.detach()[0].float().cpu()  

        if need_GT:
            out_dict["X_GT"] = self.X_GT.detach()[0].float().cpu()
            out_dict["Y_GT"] = self.Y_GT.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_AE"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)

