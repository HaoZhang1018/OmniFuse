import json
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output




class DepthwiseConv(nn.Module):


    def __init__(self, in_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False):
        super(DepthwiseConv, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=in_channels,
            bias=bias
        )

    def forward(self, x):
        out = self.conv(x)
        return out





class PointwiseConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(PointwiseConv, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out



class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)):
        super(DepthwiseSeparableConv, self).__init__()

        self.conv1 = DepthwiseConv(
            in_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )

        self.conv2 = PointwiseConv(
            in_channels=in_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out





class DownSampling(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer=None):
        super(DownSampling, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[-1] // 2)
        )

        if norm_layer is None:
            self.norm = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.norm = norm_layer

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return out


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(
            self,
            args=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 100,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(_MatrixDecomposition2DBase, self).__init__()
        args: dict = json.loads(args)
        for k, v in args.items():
            setattr(self, k, v)

    @abstractmethod
    def _build_bases(self, batch_size):
        pass

    @abstractmethod
    def local_step(self, x, bases, coef):
        pass

    @torch.no_grad()
    def local_inference(self, x, bases):
        # (batch_size * MD_S, MD_D, N)^T @ (batch_size * MD_S, MD_D, MD_R) -> (batchszie * MD_S, N, MD_R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.INV_T * coef, dim=-1)

        steps = self.TRAIN_STEPS if self.training else self.EVAL_STEPS
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    @abstractmethod
    def compute_coef(self, x, bases, coef):
        pass

    def forward(self, x):

        batch_size, channels, height, width = x.shape

        # (batch_size, channels, height, width) -> (batch_size * MD_S, MD_D, N)
        if self.SPATIAL:
            self.MD_D = channels // self.MD_S
            N = height * width
            x = x.view(batch_size * self.MD_S, self.MD_D, N)
        else:
            self.MD_D = height * width
            N = channels // self.MD_S
            x = x.view(batch_size * self.MD_S, N, self.MD_D).transpose(1, 2)

        if not self.RAND_INIT and not hasattr(self, 'bases'):
            bases = self._build_bases(1)
            self.register_buffer('bases', bases)

        # (MD_S, MD_D, MD_R) -> (batch_size * MD_S, MD_D, MD_R)
        if self.RAND_INIT:
            bases = self._build_bases(batch_size)
        else:
            bases = self.bases.repeat(batch_size, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (batch_size * MD_S, N, MD_R)
        coef = self.compute_coef(x, bases, coef)

        # (batch_size * MD_S, MD_D, MD_R) @ (batch_size * MD_S, N, MD_R)^T -> (batch_size * MD_S, MD_D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (batch_size * MD_S, MD_D, N) -> (batch_size, channels, height, width)
        if self.SPATIAL:
            x = x.view(batch_size, channels, height, width)
        else:
            x = x.transpose(1, 2).view(batch_size, channels, height, width)

        # (batch_size * height, MD_D, MD_R) -> (batch_size, height, N, MD_D)
        bases = bases.view(batch_size, self.MD_S, self.MD_D, self.MD_R)

        if self.return_bases:
            return x, bases
        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(
            self,
            args=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(NMF2D, self).__init__(args)

    def _build_bases(self, batch_size):
        bases = torch.rand((batch_size * self.MD_S, self.MD_D, self.MD_R)).to(self.device)
        bases = F.normalize(bases, dim=1)

        return bases

    # @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (batch_size * MD_S, MD_D, N)^T @ (batch_size * MD_S, MD_D, MD_R) -> (batch_size * MD_S, N, MD_R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (batch_size * MD_S, N, MD_R) @ [(batch_size * MD_S, MD_D, MD_R)^T @ (batch_size * MD_S, MD_D, MD_R)]
        # -> (batch_size * MD_S, N, MD_R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (batch_size * MD_S, MD_D, N) @ (batch_size * MD_S, N, MD_R) -> (batch_size * MD_S, MD_D, MD_R)
        numerator = torch.bmm(x, coef)
        # (batch_size * MD_S, MD_D, MD_R) @ [(batch_size * MD_S, N, MD_R)^T @ (batch_size * MD_S, N, MD_R)]
        # -> (batch_size * MD_S, D, MD_R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (batch_size * MD_S, MD_D, N)^T @ (batch_size * MD_S, MD_D, MD_R) -> (batch_size * MD_S, N, MD_R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (batch_size * MD_S, N, MD_R) @ (batch_size * MD_S, MD_D, MD_R)^T @ (batch_size * MD_S, MD_D, MD_R)
        # -> (batch_size * MD_S, N, MD_R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)
        return coef