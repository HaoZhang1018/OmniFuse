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

from models.modules.loss import fusion_base_loss,fusion_modulated_loss
import random
from .base_model import BaseModel
from  models.modules.mix_segnext import SegNeXt
import torchvision.transforms.functional as TF
from torchvision import io
logger = logging.getLogger("base")

# segment anything
from segment_anything import sam_model_registry, SamPredictor
from .seg_util import load_image, generation_promotlist_for_train,generation_promotlist_for_train1, compute_dice_loss_with_textlist, get_grounding_output, load_model, tensor2img, calculate_clip
import cv2


def generation_promotlist_for_train_MFNet(
    seg_label_tensor, 
    classes_list=["Background", "Cars", "Pedestrians", "Bikes", "Curve", "Car Stop", "Guardrail", "Color cone", "Bump"]
):
    """
    Generate a list of formatted prompts and target segment lists.

    Parameters:
        classes_list (list): List of class names with 'Background' as the first element.
        seg_label_tensor (torch.Tensor): 4D tensor of shape [batch_size, C, H, W] representing segmentation labels.

    Returns:
        tuple: (formatted_promot_list, text_seg_list)
    """
    # Ensure seg_label_tensor is a 4D tensor
    if not isinstance(seg_label_tensor, torch.Tensor) or len(seg_label_tensor.shape) != 4:
        raise ValueError("seg_label_tensor must be a 4D tensor with shape [batch_size, C, H, W].")
    
    # Define the specific target classes to select from
    specific_target_classes = [1, 2, 3]  # Corresponding to Car, Person, Bike, Color Cone
    
    # Initialize lists
    text_seg_list = []
    formatted_promot_list = []
    
    batch_size = seg_label_tensor.shape[0]

    for b in range(batch_size):
        # Get unique pixel values for the current batch
        unique_values = torch.unique(seg_label_tensor[b].flatten()).tolist()

        # Convert to set for faster membership checking
        unique_values_set = set(unique_values)

        # Find available target classes
        available_target_classes = [c for c in specific_target_classes if c in unique_values_set]

        if available_target_classes:
            # Select only one class randomly from available target classes
            selected_class = random.choice(available_target_classes)

            # Create segment list with background (0) and selected class
            seg_list = [0, selected_class]  # Include background and selected class
        else:
            seg_list = [0]  # Only background

        text_seg_list.append(seg_list)

        # Generate formatted prompts based on selected classes
        if len(seg_list) > 1:
            selected_class_names = [classes_list[c] for c in seg_list[1:] if c < len(classes_list)]
            highlight_string = ' and '.join(selected_class_names)
            highlight_string="I am interested in "+ highlight_string
        else:
            highlight_string = "\n"
        formatted_promot_list.append(highlight_string)
    return formatted_promot_list, text_seg_list

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
                self.Seg_model = SegNeXt(num_classes=9, device=self.device)
                checkpoint = torch.load("pretrained/best.pth", map_location='cpu')
                state_dict = checkpoint.get('state_dict', checkpoint)
                new_backbone_state = {}
                new_head_state = {}

                for k, v in state_dict.items():
                    if k.startswith("backbone."):
                        new_backbone_state[k.replace("backbone.", "")] = v
                    elif k.startswith("decode_head."):
                        new_head_state[k.replace("decode_head.", "")] = v

                missing1, unexpected1 = self.Seg_model.backbone.load_state_dict(new_backbone_state, strict=True)
                missing2, unexpected2 = self.Seg_model.decode_head.load_state_dict(new_head_state, strict=True)
                for param in self.Seg_model.parameters():
                    param.requires_grad = True
                print("Backbone Missing:", missing1)
                print("Backbone Unexpected:", unexpected1)
                print("Decode Head Missing:", missing2)
                print("Decode Head Unexpected:", unexpected2)
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
            if opt["Fusion_Model_type"]=='modulated':
                self.Seg_model = DistributedDataParallel(self.Seg_model, device_ids=[torch.cuda.current_device()]) 
        self.load()

        self.encode = self.ae_model.encode
        self.decode = self.ae_model.decode
        
        self.diff_model_X.eval()
        self.diff_model_Y.eval()

        if self.is_train:
            self.fuse_model.train()
            if opt["Fusion_Model_type"]=='modulated':
                self.Seg_model.train()

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
            if opt["Fusion_Model_type"]=='modulated':
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
        mask_all=torch.zeros((self.Modal_X.shape[0], self.Modal_X.shape[2], self.Modal_X.shape [3])).to(self.device)
        self.text_list=[]
        print(f"use {Fusion_Model_type} model to infer")
        if Fusion_Model_type == "modulated":
            self.Modal_X_M=(self.Modal_X/(torch.max(self.Modal_X)-torch.min(self.Modal_X)))
            self.Modal_Y_M=(self.Modal_Y/(torch.max(self.Modal_Y)-torch.min(self.Modal_Y)))
            for index, text_prompt in enumerate(prompt):
                text_prompt = text_prompt.replace('\n', '')
                if text_prompt=='' and self.is_train:
                    text_prompt , text_id = generation_promotlist_for_train_MFNet(self.Seg_Label[index:index+1])
                    self.text_list.append(text_id)
                else:
                    pass
                if isinstance(text_prompt, list):
                    text_prompt=text_prompt[0]
                if text_prompt=='' or text_prompt=='\n':
                    print("no prompt detect, will not process the prompt")
                    print("please check your config or inputs")
                else:
                    print(f"use prompt: {text_prompt}")
                    box_threshold = 0.25
                    text_threshold = 0.25

                    #---------------------------------------------#
                    image = TF.normalize(self.Modal_X_M[index], mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
                    # run grounding dino model
                    boxes_filt, pred_phrases = get_grounding_output(
                        self.GroundingDINO_model, image, text_prompt, box_threshold, text_threshold, device=self.device
                    )
                    # initialize SAM
                    image = ((self.Modal_X_M[index].cpu().permute(1, 2, 0))* 255).numpy().astype(np.uint8)
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
                            mask_all[index:index+1]=torch.max(mask_all[index:index+1],masks[idx])
                
                    #---------------------------------------------#
                    image = TF.normalize(self.Modal_Y_M[index], mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
                     # run grounding dino model
                    boxes_filt, pred_phrases = get_grounding_output(
                        self.GroundingDINO_model, image, text_prompt, box_threshold, text_threshold, device=self.device
                    )
                    # initialize SAM
                    image = ((self.Modal_Y_M[index].cpu().permute(1, 2, 0))* 255).numpy().astype(np.uint8)
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
                            mask_all[index:index+1]=torch.max(mask_all[index:index+1], masks[idx])
        self.Seg_Label_vision=mask_all
        self.text_embedding=mask_all.unsqueeze(1)

    def Seg_target(self,image_Seg_Label, text_seg_list):
        B, W, H = image_Seg_Label.shape
  
        if len(text_seg_list) != B:
            raise ValueError("text_seg_list length must match the batch dimension of tensors.")
    
        Seg_Label_C = image_Seg_Label.clone()
    
        for b in range(B):
            sublist = text_seg_list[b] 
            if not sublist: 
                continue
      
            sublist_tensor = torch.tensor(sublist, dtype=torch.long, device=self.device)
        
            mask = torch.isin(image_Seg_Label[b], sublist_tensor)
    
            Seg_Label_C[b][~mask] = 0
    
        return Seg_Label_C

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
            Fuse_M = (Fuse - torch.min(Fuse)) / (torch.max(Fuse) - torch.min(Fuse) + 1e-6)
            mean = torch.tensor([0.485, 0.456, 0.406], device=Fuse_M.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=Fuse_M.device).view(1, 3, 1, 1)
            Fuse_M = (Fuse_M - mean) / std
            Fuse_segmap=self.Seg_model(Fuse_M)
            self.Seg_Label_C = self.Seg_target(self.Seg_Label[:,0,:,:], self.text_list)
            total_loss=self.loss_fn(self.Modal_X, self.Modal_Y, Fuse,Fuse_segmap,self.Seg_Label[:,0,:,:],self.Fusion_Base,self.Seg_Label_C) 
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
