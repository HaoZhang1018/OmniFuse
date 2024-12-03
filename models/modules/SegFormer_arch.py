import torch
import torch.nn as nn
import torch.nn.functional as F 
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from torch.utils.data import DataLoader
from functools import partial
from . import mix_transformer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from PIL import Image
import math
import numpy as np

from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import os
import shutil







class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides=None, in_channels=128, embedding_dim=256, num_classes=20, **kwargs):
        super(SegFormerHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        #decoder_params = kwargs['decoder_params']
        #embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.dropout = nn.Dropout2d(0.1)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x):

        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x















class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.backbone = backbone
        self.feature_strides = [4, 8, 16, 32]
        #self.in_channels = [32, 64, 160, 256]
        #self.in_channels = [64, 128, 320, 512]

        self.encoder = getattr(mix_transformer, backbone)()
        self.in_channels = self.encoder.embed_dims
        ## initilize encoder
        if pretrained:
            print('pretrained/'+backbone+'.pth')
            state_dict = torch.load('pretrained/'+backbone+'.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict,)

        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        
        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes, kernel_size=1, bias=False)

    def initialize(self,):
        state_dict = torch.load('pretrained/' + self.backbone + '.pth')
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        self.encoder.load_state_dict(state_dict, )
    def _forward_cam(self, x):
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)
        
        return cam

    def get_param_groups(self):

        param_groups = [[], [], []] # 
        
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):

            param_groups[2].append(param)
        
        param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, x):

        _x = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        cls = self.classifier(_x4)

        return self.decoder(_x)
        

class SegFormer(nn.Module):

    def __init__(self,backbone, num_classes=20, embedding_dim=256, pretrained=True):
        super(SegFormer, self).__init__()

        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        self.denoise_net = WeTr(backbone,num_classes,embedding_dim,pretrained)
        # self.discriminator = Discriminator()
        self.mean =[123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
    def forward(self, fused_seg1):

        torch_norma = fused_seg1*255
        for index in range(3):
            torch_norma[:,index,:,:] = (torch_norma[:,index,:,:] - self.mean[index])/self.std[index]

        seg_map = self.denoise_net(torch_norma)
        return seg_map

    def _loss(self,fused_seg1,label,criterion):
        torch_norma = fused_seg1 * 255
        for index in range(3):
            torch_norma[:, index, :, :] = (torch_norma[:, index, :, :] - self.mean[index]) / self.std[index]
        seg_map = self.denoise_net(torch_norma)
        outputs = F.interpolate(seg_map, size=label.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = criterion(outputs,label.type(torch.long))
        return denoise_loss


    def enhance_net_parameters(self):
        return self.enhance_net.parameters()

    def denoise_net_parameters(self):
        return self.denoise_net.parameters()


