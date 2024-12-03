import os
import random
import sys
import torch.nn.functional as F

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data
import math
try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass


class XYPromptDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    And Corresponding prompt.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.Modal_X_paths, self.Modal_Y_paths = None, None
        self.Modal_X_env, self.Modal_Y_env = None, None  # environment for lmdb
        if self.opt["phase"] == "train" and self.opt["With_prompt"] == True:
            self.Seg_Label_paths, self.Fusion_Base_paths = None, None
            self.Seg_Label_env, self.Fusion_Base_env = None, None  # environment for lmdb
        self.Modal_X_size, self.Modal_Y_size = opt["Modal_X_size"], opt["Modal_Y_size"]

        # read image list from lmdb or image files
        if opt["data_type"] == "img":
            self.Modal_X_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_X"]
            )  # Modal_X list
            self.Modal_Y_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_Y"]
            )  # Modal_Y list
            if self.opt["phase"] == "train" and self.opt["With_prompt"] == True:
                self.Seg_Label_paths = util.get_image_paths(
                    opt["data_type"], opt["dataroot_Seg_Label"]
                )  # Seg_Label list
                self.Fusion_Base_paths = util.get_image_paths(
                    opt["data_type"], opt["dataroot_Fusion_Base"]
                )  # Fusion_Base list
        else:
            print("Error: data_type is not matched in Dataset")
        
        assert self.Modal_X_paths, "Error: Modal_X paths are empty."
        assert self.Modal_Y_paths, "Error: Modal_Y paths are empty."
        if self.opt["phase"] == "train" and self.opt["With_prompt"] == True:
            assert self.Seg_Label_paths, "Error: Seg_Label paths are empty."
            assert self.Fusion_Base_paths, "Error: Fusion_Base paths are empty."
        if self.Modal_X_paths and self.Modal_Y_paths:
            assert len(self.Modal_X_paths) == len(
                self.Modal_Y_paths
            ), "GT and LR datasets have different number of images - {}, {}.".format(
                len(self.Modal_X_paths), len(self.Modal_Y_paths)
            )
        if self.opt["With_prompt"] == True:
            with open(opt["prompt_list"], "r") as f:
                 self.prompt_list = f.readlines()
            if opt["prompt_list_X"] is not None:
                with open(opt["prompt_list_X"], "r") as f:
                    self.prompt_list_X = f.readlines()
            if opt["prompt_list_Y"] is not None:
                with open(opt["prompt_list_Y"], "r") as f:
                    self.prompt_list_Y = f.readlines()
        
            self.random_scale_list = [1]

    def __getitem__(self, index):
        Modal_X_path, Modal_Y_path = None, None
        if self.opt["phase"] == "train" and self.opt["With_prompt"] == True:
            Seg_Label_path, Fusion_Base_path = None, None
        scale = self.opt["scale"] if self.opt["scale"] else 1
        Modal_X_size = self.opt["Modal_X_size"]
        Modal_Y_size = self.opt["Modal_Y_size"]

        # get Modal_X image
        Modal_X_path = self.Modal_X_paths[index]
        Modal_X = util.read_img(
            self.Modal_X_env, Modal_X_path, size=None
        )  # return: Numpy float32, HWC, BGR, [0,1]
        # modcrop in the validation / test phase
        Initial_H,Initial_W,_=Modal_X.shape
        if self.opt["phase"] != "train":
            Modal_X = util.modcrop(Modal_X, scale)

        # get Modal_Y image
        Modal_Y_path = self.Modal_Y_paths[index]
        Modal_Y = util.read_img(
            self.Modal_Y_env, Modal_Y_path, size=None
        )  # return: Numpy float32, HWC, BGR, [0,1]
        if self.opt["phase"] == "train" and self.opt["With_prompt"] == True:
            # get Seg_Label image
            Seg_Label_path = self.Seg_Label_paths[index]
            Seg_Label = util.read_Label(
                self.Seg_Label_env, Seg_Label_path, size=None
            )  # return: Numpy float32, HWC, BGR, [0,1]     
            # get Fusion_Base image
            Fusion_Base_path = self.Fusion_Base_paths[index]
            Fusion_Base = util.read_img(
                self.Fusion_Base_env, Fusion_Base_path, size=None
            )  # return: Numpy float32, HWC, BGR, [0,1]   
        pad_h = math.ceil(Modal_X.shape[0] / 64) * 64 - Modal_X.shape[0]
        pad_w = math.ceil(Modal_X.shape[1] / 64) * 64 - Modal_X.shape[1]
        Modal_X = np.pad(Modal_X, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        Modal_Y = np.pad(Modal_Y, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        if self.opt["phase"] == "train" and self.opt["With_prompt"] == True:
            Seg_Label = np.pad(Seg_Label, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            Fusion_Base = np.pad(Fusion_Base, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

        # modcrop in the validation / test phase
        if self.opt["phase"] != "train":
            #Modal_Y = util.modcrop(Modal_Y, scale)
            Modal_X = Modal_X
            Modal_Y = Modal_Y
        
        if self.opt["phase"] == "train":
            # process images for training
            H, W, C = Modal_X.shape
            assert Modal_X_size == Modal_Y_size, "Modal_X size does not match Modal_Y size"
            Modal_X = Modal_X
            Modal_Y = Modal_Y
            if self.opt["With_prompt"] == True:
                Seg_Label = Seg_Label
                Fusion_Base = Fusion_Base

            # augmentation - flip, rotate
                Modal_X, Modal_Y, Seg_Label, Fusion_Base = util.augment(
                    [Modal_X, Modal_Y, Seg_Label, Fusion_Base],
                    self.opt["use_flip"],
                    self.opt["use_rot"],
                    self.opt["mode"],
                    self.opt["use_swap"],
                )
            else:
                Modal_X, Modal_Y = util.augment(
                    [Modal_X, Modal_Y],
                    self.opt["use_flip"],
                    self.opt["use_rot"],
                    self.opt["mode"],
                    self.opt["use_swap"],
                )
        
        elif Modal_X_size is not None:
            H, W, C = Modal_X.shape
            assert Modal_X_size == Modal_Y_size, "Modal_X size does not match Modal_Y size"

            if Modal_X_size < H and Modal_X_size < W:
                rnd_h = H // 2 - Modal_X_size//2
                rnd_w = W // 2 - Modal_X_size//2
                Modal_X = Modal_X[rnd_h : rnd_h + Modal_X_size, rnd_w : rnd_w + Modal_X_size, :]
                Modal_Y = Modal_Y[rnd_h : rnd_h + Modal_X_size, rnd_w : rnd_w + Modal_X_size, :]

        # change color space if necessary
        if self.opt["color"]:
            Modal_X = util.channel_convert(Modal_X.shape[2], self.opt["color"], [Modal_X])[
                0
            ]  # TODO during val no definition
            Modal_Y = util.channel_convert(Modal_Y.shape[2], self.opt["color"], [Modal_Y])[
                0
            ]
            if self.opt["phase"] == "train" and self.opt["With_prompt"] == True:
                Seg_Label = util.channel_convert(Seg_Label.shape[2], self.opt["color"], [Seg_Label])[
                    0
                ]
                Fusion_Base = util.channel_convert(Fusion_Base.shape[2], self.opt["color"], [Fusion_Base])[
                    0
                ]

        if Modal_X.shape[2] == 1:
            Modal_X = np.dstack((Modal_X, Modal_X, Modal_X))

        if Modal_Y.shape[2] == 1:
            Modal_Y = np.dstack((Modal_Y, Modal_Y, Modal_Y))
            
        if self.opt["phase"] == "train" and self.opt["With_prompt"] == True:
            if Seg_Label.shape[2] == 1:
                Seg_Label = np.dstack((Seg_Label, Seg_Label, Seg_Label))

            if Fusion_Base.shape[2] == 1:
                Fusion_Base = np.dstack((Fusion_Base, Fusion_Base, Fusion_Base))

        # BGR to RGB, HWC to CHW, numpy to tensor
        if Modal_X.shape[2] == 3:
            Modal_X = Modal_X[:, :, [2, 1, 0]]                   
        Modal_X = torch.from_numpy(
            np.ascontiguousarray(np.transpose(Modal_X, (2, 0, 1)))
        ).float()

        if Modal_Y.shape[2] == 3:
            Modal_Y = Modal_Y[:, :, [2, 1, 0]]   
        Modal_Y = torch.from_numpy(
            np.ascontiguousarray(np.transpose(Modal_Y, (2, 0, 1)))
        ).float()

        if self.opt["phase"] == "train" and  self.opt["With_prompt"] == True:
            if Seg_Label.shape[2] == 3:
                Seg_Label = Seg_Label[:, :, [2, 1, 0]]
            Seg_Label = torch.from_numpy(
                np.ascontiguousarray(np.transpose(Seg_Label, (2, 0, 1)))
            ).float()

            if Fusion_Base.shape[2] == 3:
                Fusion_Base = Fusion_Base[:, :, [2, 1, 0]]
            Fusion_Base = torch.from_numpy(
                np.ascontiguousarray(np.transpose(Fusion_Base, (2, 0, 1)))
            ).float()
        if self.opt["With_prompt"] == True:
            prompt = self.prompt_list[index]
        Modal_X = F.pad(Modal_X, (0, 64, 0, 64), mode='reflect')
        Modal_Y = F.pad(Modal_Y, (0, 64, 0, 64), mode='reflect')
        if self.opt["phase"] == "train" and self.opt["With_prompt"] == True:
            Seg_Label = F.pad(Seg_Label, (0, 64, 0, 64), mode='reflect')
            Fusion_Base = F.pad(Fusion_Base, (0, 64, 0, 64), mode='reflect')
        if self.opt["phase"] == "train" and self.opt["With_prompt"] == False:
            return {
                "Modal_X": Modal_X,  "Modal_X_path": Modal_X_path, 
                "Modal_Y": Modal_Y,  "Modal_Y_path": Modal_Y_path,
                "Initial_H": Initial_H, "Initial_W": Initial_W
                }
        elif self.opt["phase"] == "train" and self.opt["With_prompt"] == True:
            return{
                "Modal_X": Modal_X, "Modal_X_path": Modal_X_path,
                "Modal_Y": Modal_Y, "Modal_Y_path": Modal_Y_path,
                "Seg_Label":Seg_Label,"Fusion_Base":Fusion_Base,
                "prompt": prompt,
                "Initial_H": Initial_H, "Initial_W": Initial_W
            }
        elif self.opt["phase"] != "train" and self.opt["With_prompt"] == True:
            return{
                "Modal_X": Modal_X, "Modal_X_path": Modal_X_path,
                "Modal_Y": Modal_Y, "Modal_Y_path": Modal_Y_path,
                "prompt": prompt,
                "Initial_H": Initial_H, "Initial_W": Initial_W
            }
        else:
            return{
                "Modal_X": Modal_X, "Modal_X_path": Modal_X_path,
                "Modal_Y": Modal_Y, "Modal_Y_path": Modal_Y_path,
                "Initial_H": Initial_H, "Initial_W": Initial_W
            }
        
    def __len__(self):
        return len(self.Modal_X_paths)