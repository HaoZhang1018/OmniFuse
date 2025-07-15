
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from .model import MSCAN, LightHamHead

class SegNeXt(nn.Module):
    def __init__(self,
                 device='cpu',
                 num_classes=9,
                 backbone_cfg=dict(
                     in_chans=3,
                     embed_dims=[64, 128, 320, 512],
                     mlp_ratios=[8, 8, 4, 4],
                     depths=[3, 3, 12, 3],
                     drop_path_rate=0.1,
                     norm_cfg=dict(type='SyncBN', requires_grad=True)
                 ),
                 head_cfg=dict(
                     in_channels=[128, 320, 512],
                     channels=512,
                     ham_channels=512,
                     norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                     align_corners=False,
                 )):
        super().__init__()
        self.device = device
        self.backbone = MSCAN(**backbone_cfg).to(self.device)
        self.decode_head = LightHamHead(
            in_channels=head_cfg['in_channels'],
            in_index=[1, 2, 3],
            channels=head_cfg['channels'],
            ham_channels=head_cfg['ham_channels'],
            num_classes=num_classes,
            norm_cfg=head_cfg['norm_cfg'],
            align_corners=head_cfg['align_corners'],
            device=self.device
        ).to(self.device)

    def forward(self, x):
        feats = self.backbone(x)
        out = self.decode_head(feats)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out
