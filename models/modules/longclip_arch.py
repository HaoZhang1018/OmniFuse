import torch
import torch.nn as nn
from einops import rearrange, repeat
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC

from .LongCLIP.model import longclip

class FrozenCLIPEmbedder(nn.Module):
    def __init__(self, version='ViT-L/14', device="cuda", n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = longclip.load(version, device=device)
        self.device = device
        self.n_repeat = n_repeat
        self.normalize = normalize
        self.input_resolution = self.model.visual.input_resolution
        print("long clip image resolution: ", self.input_resolution)
        self._transforms = Compose([
            Resize(self.input_resolution, interpolation=BICUBIC),
            CenterCrop(self.input_resolution),
            # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        self.model = self.model.to(torch.float32)
        # self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def preprocess(self, image):
        image = self._transforms(image)
        return image

    def encode_image(self, image):
        image = self.preprocess(image)
        image_features = self.model.encode_image(image)
        if self.normalize:
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features
    
    def encode_text(self, text):
        tokens = longclip.tokenize(text).to(self.device)
        text_features = self.model.encode_text(tokens)
        if self.normalize:
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # for SpatialTransformer
        if text_features.ndim==2:
            text_features = text_features[:, None, :]
        text_features = repeat(text_features, 'b 1 d -> b k d', k=self.n_repeat)
        return text_features
    
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text
        