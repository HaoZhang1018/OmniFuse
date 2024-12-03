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
import utils as util
# from ema_pytorch import EMA
from einops import rearrange
import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
##
import torch
import argparse
import yaml
import math
import os
import time
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F

from math import ceil
import numpy as np
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from monai.losses import DiceLoss

from models.modules.loss import fusion_base_loss,fusion_modulated_loss

from .base_model import BaseModel
from  models.modules.mix_segnext import SegNeXt_B
import torchvision.transforms.functional as TF
from torchvision import io
logger = logging.getLogger("base")

# segment anything
from segment_anything import sam_model_registry, SamPredictor
from .seg_util import load_image, generation_promotlist_for_train,generation_promotlist_for_train1, compute_dice_loss_with_textlist, get_grounding_output, load_model, tensor2img, calculate_clip



class FusionModel(BaseModel):
    def __init__(self, opt):
        super(FusionModel, self).__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.diff_model_X = networks.define_Diff(opt).to(self.device)
        self.diff_model_Y = networks.define_Diff(opt).to(self.device)
        self.ae_model = networks.define_AE(opt).to(self.device)
        self.fuse_model = networks.define_Fuse(opt).to(self.device)
        if opt["Fusion_Model_type"]=='modulated':
            if opt["Fusion_task"]=='train':
                self.Seg_model = SegNeXt_B(num_classes=9, pretrained_cfg="/pretrained/mscan_b.pth")
                self.Seg_model.load_state_dict(torch.load("/pretrained/best.pth"))
            self.GroundingDINO_model = load_model(opt["Grounding_SAM"]["config_file_path"], opt["Grounding_SAM"]["grounded_checkpoint_path"], device=self.device)   # load groundingDINO
            self.SAM_model =SamPredictor(sam_model_registry[opt["Grounding_SAM"]["sam_version"]](checkpoint=opt["Grounding_SAM"]["sam_checkpoint_path"]).to(device=self.device))    
        
        
        for param in self.ae_model.parameters():
            param.requires_grad = False
                
        for param in self.diff_model_X.parameters():
            param.requires_grad = False                

        for param in self.diff_model_Y.parameters():
            param.requires_grad = False   
        
                              
        if opt["dist"]:
            self.fuse_model = DistributedDataParallel(self.fuse_model, device_ids=[torch.cuda.current_device()])

        self.load()

        self.encode = self.ae_model.encode
        self.decode = self.ae_model.decode
        
        self.diff_model_X.eval()
        self.diff_model_Y.eval()

        if self.is_train:
            self.fuse_model.train()

            if opt["Fusion_Model_type"]=='modulated':
                self.loss_fn = fusion_modulated_loss().to(self.device)
            if opt["Fusion_Model_type"]=='base':
                self.loss_fn = fusion_base_loss().to(self.device)

            self.weight = opt['train']['weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.fuse_model.named_parameters():  # can optimize for a part of the model
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

            self.log_dict = OrderedDict()
    
    def add_dataset(self,Seg_Label=None, Fusion_Base=None):           
        if Seg_Label==None or Fusion_Base==None:
            self.Seg_Label=None
            self.Fusion_Base=None
        else:
            self.Seg_Label = Seg_Label.to(self.device) 
            self.Fusion_Base = Fusion_Base.to(self.device) 

    def feed_data(self, Modal_X_fea, Modal_Y_fea, Fusion_Model_type='base', prompt=None):

        self.Modal_latent_X = Modal_X_fea[0].to(self.device) # latent
        self.Modal_latent_Y = Modal_Y_fea[0].to(self.device) # latent
        
        # get hidden state using Mean Fusion Rule
        self.hidden_X = []
        self.hidden_Y = []
        self.hidden_fuse = []
        for i in range(1, len(Modal_X_fea)):
            hidden_fuse = (Modal_X_fea[i] + Modal_Y_fea[i]) / 2
            self.hidden_fuse.append(hidden_fuse.to(self.device))
            self.hidden_X.append(Modal_X_fea[i].to(self.device))
            self.hidden_Y.append(Modal_Y_fea[i].to(self.device))
            
        self.Modal_X = self.decode(self.Modal_latent_X, self.hidden_X)
        self.Modal_Y = self.decode(self.Modal_latent_Y, self.hidden_Y)
        
        #-----------------------------------#
        mask_all=torch.zeros((1, self.Modal_X.shape[2], self.Modal_X.shape [3])).to(self.device)
        print(f"use {Fusion_Model_type} model to infer")
        if Fusion_Model_type == "modulated":
            text_prompt = prompt
            
            if isinstance(text_prompt, list):
                text_prompt = text_prompt[0]
            
            text_prompt = text_prompt.replace('\n', '')
            
            if text_prompt=='':
                print("no prompt detect, will not process the prompt")
                print("please check your config or inputs")
            else:
                print(f"use prompt: {text_prompt}")
                box_threshold = 0.25
                text_threshold = 0.25

                #---------------------------------------------#
                self.Modal_X_M=(self.Modal_X/(torch.max(self.Modal_X)-torch.min(self.Modal_X)))
                image = TF.normalize(self.Modal_X_M.squeeze(0), mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
                # run grounding dino model
                boxes_filt, pred_phrases = get_grounding_output(
                    self.GroundingDINO_model, image, text_prompt, box_threshold, text_threshold, device=self.device
                )
                # initialize SAM
                image = ((self.Modal_X_M.cpu().squeeze(0).permute(1, 2, 0))* 255).numpy().astype(np.uint8)
                self.SAM_model.set_image(image)
                size = (image.shape[1],image.shape[0])
                H, W = size[1], size[0]
                if boxes_filt.size(0)>0:
                    for i in range(boxes_filt.size(0)):
                        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                        boxes_filt[i][2:] += boxes_filt[i][:2]

                    boxes_filt = boxes_filt.cpu()
                    transformed_boxes = self.SAM_model.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

                    masks, _, _ = self.SAM_model.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes.to(self.device),
                        multimask_output=False,
                    )
        
                    for idx in range(masks.shape[0]):
                        mask_all=torch.max(mask_all,masks[idx])
                
                #---------------------------------------------#
                self.Modal_Y_M=(self.Modal_Y/(torch.max(self.Modal_Y)-torch.min(self.Modal_Y)))
                image = TF.normalize(self.Modal_Y_M.squeeze(0), mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
                # run grounding dino model
                boxes_filt, pred_phrases = get_grounding_output(
                    self.GroundingDINO_model, image, text_prompt, box_threshold, text_threshold, device=self.device
                )
                # initialize SAM
                image = ((self.Modal_Y_M.cpu().squeeze(0).permute(1, 2, 0))* 255).numpy().astype(np.uint8)
                self.SAM_model.set_image(image)

                size = (image.shape[1],image.shape[0])
                H, W = size[1], size[0]
                if boxes_filt.size(0)>0:
                    for i in range(boxes_filt.size(0)):
                        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                        boxes_filt[i][2:] += boxes_filt[i][:2]

                    boxes_filt = boxes_filt.cpu()
                    transformed_boxes = self.SAM_model.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

                    masks, _, _ = self.SAM_model.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes.to(self.device),
                        multimask_output=False,
                    )
            
                    for idx in range(masks.shape[0]):
                        mask_all=torch.max(mask_all, masks[idx])
                        
        self.Seg_Label_vision=mask_all
        self.text_embedding=mask_all.unsqueeze(0)

    def keep_class(self,g_label, image_Seg_Label, text_seg_list):
        B, W, H = g_label.shape
  
        if len(text_seg_list) != B:
            raise ValueError("text_seg_list length must match the batch dimension of tensors.")
    
        new_g_label = g_label.clone()
        new_image_Seg_Label = image_Seg_Label.clone()
    
        for b in range(B):
            sublist = text_seg_list[b] 
            if not sublist: 
                continue
      
            sublist_tensor = torch.tensor(sublist, dtype=torch.long, device=self.device)
        
            mask = torch.isin(g_label[b], sublist_tensor)
            mask1 = torch.isin(image_Seg_Label[b], sublist_tensor)
    
            new_g_label[b][~mask] = 0
            new_image_Seg_Label[b][~mask1] = 0
    
        return new_g_label, new_image_Seg_Label

    def optimize_parameters(self, step,Fusion_Model_type='base'):
        self.optimizer.zero_grad()

        if self.opt["dist"]:
            fuse_fn = self.fuse_model.module
        else:
            fuse_fn = self.fuse_model
        latent_fuse = fuse_fn(self.Modal_latent_X, self.Modal_latent_Y, context=self.text_embedding)
        hidden_fuse = self.hidden_fuse
        
        # first decode latent to image
        Fuse = self.decode(latent_fuse, hidden_fuse)
        if Fusion_Model_type == 'base':
            total_loss, loss_int,loss_grad, loss_color=self.loss_fn(self.Modal_X, self.Modal_Y, Fuse) 
            total_loss.backward()
            self.optimizer.step()
        
            self.log_dict["total_loss"] = total_loss.item()
            self.log_dict["loss_int"] = loss_int.item()
            self.log_dict["loss_grad"] = loss_grad.item()
            self.log_dict["loss_color"] = loss_color.item()
      
        else:
            Fuse_M=Fuse / (torch.max(Fuse)-torch.min(Fuse))
            Fuse_segmap=self.Seg_model.forward(Fuse_M)
            seg_output = F.interpolate(Fuse_segmap, size=(Fuse_M.squeeze(0)).shape[1:], mode='bilinear', align_corners=False).softmax(dim=1)

            total_loss=self.loss_fn(self.Modal_X, self.Modal_Y, Fuse,seg_output,self.Seg_Label[:,0:1,:,:],self.Fusion_Base,self.text_seg_list) 
            total_loss.backward()
            self.optimizer.step()
            self.log_dict["total_loss"] = total_loss.item()
        
    def test(self):
        self.fuse_model.eval()

        if self.opt["dist"]:
            fuse_fn = self.fuse_model.module
        else:
            fuse_fn = self.fuse_model
        with torch.no_grad():
            latent_fuse = fuse_fn(self.Modal_latent_X, self.Modal_latent_Y, context=self.text_embedding)
            hidden_fuse = self.hidden_fuse
            self.output = self.decode(latent_fuse, hidden_fuse)

                   
        self.fuse_model.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()

        out_dict["Output"] = self.output.detach()[0].float().cpu()

        return out_dict



    def load(self):
        print("start to load pretrained model...")
        load_path_Diff_X = self.opt["path"]["pretrain_model_Diff_X"]
        if load_path_Diff_X is not None:
            logger.info("Loading model for LatentDiffusion_X [{:s}] ...".format(load_path_Diff_X))
            self.load_network(load_path_Diff_X, self.diff_model_X, self.opt["path"]["strict_load"])

        load_path_Diff_Y = self.opt["path"]["pretrain_model_Diff_Y"]
        if load_path_Diff_Y is not None:
            logger.info("Loading model for LatentDiffusion_Y [{:s}] ...".format(load_path_Diff_Y))
            self.load_network(load_path_Diff_Y, self.diff_model_Y, self.opt["path"]["strict_load"])


        load_path_AE = self.opt["path"]["pretrain_model_AE"]
        if load_path_AE is not None:
            logger.info("Loading model for AutoEncoder [{:s}] ...".format(load_path_AE))
            self.load_network(load_path_AE, self.ae_model, self.opt["path"]["strict_load"])

        load_path_Fuse = self.opt["path"]["pretrain_model_Fuse"]
        if load_path_Fuse is not None:
            logger.info("Loading model for FuseModel [{:s}] ...".format(load_path_Fuse))
            self.load_network(load_path_Fuse, self.fuse_model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.fuse_model, "fuse", iter_label)