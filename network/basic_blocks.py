import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlocks:
    @staticmethod
    def lin(ni, nf, act=nn.SiLU, norm=None, bias=True):
        layers = nn.Sequential()
        if norm:
            layers.append(norm(ni))
        if act:
            layers.append(act())
        layers.append(nn.Linear(ni, nf, bias=bias))
        return layers

    @staticmethod
    def pre_conv(ni, nf, ks=3, stride=1, act=nn.SiLU, norm=None, bias=True):
        layers = nn.Sequential()
        if norm:
            layers.append(norm(num_channels=ni))
        if act:
            layers.append(act())
        layers.append(nn.Conv2d(ni, nf, stride=stride, kernel_size=ks, padding=ks // 2, bias=bias))
        return layers

    @staticmethod
    def timestep_embedding(t, emb_dim, max_period=10000):
        exponent = -math.log(max_period) * torch.linspace(0, 1, emb_dim // 2, device=t.device)
        emb = t[:, None].float() * exponent.exp()[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return F.pad(emb, (0, 1, 0, 0)) if emb_dim % 2 == 1 else emb

    @staticmethod
    def upsample(nf):
        return nn.Sequential(nn.Upsample(scale_factor=2.0), nn.Conv2d(nf, nf, 3, padding=1))
