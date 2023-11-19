from functools import partial

import fastcore.utils as fc
import torch
import torch.nn as nn

from network.attention_blocks import CrossAttention, SelfAttentionPose
from network.basic_blocks import BasicBlocks
from utils.utils import Utils


class FilmBlock(nn.Module):
    def __init__(self, n_emb, ni, act=nn.SiLU):
        super().__init__()
        self.emb_proj = nn.Linear(n_emb, ni * 2)
        self.act = act()

    def forward(self, x, cond_embed):
        emb = self.act(cond_embed)
        emb = self.emb_proj(emb)[:, :, None, None]
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = x * (1 + scale) + shift
        return x


class ResBlock(nn.Module):
    def __init__(self, n_emb, n_pose, ni, nf=None, img_hw=None, ks=3, act=nn.SiLU, norm=None, attn_chans=0):
        super().__init__()
        if nf is None:
            nf = ni

        if norm is None:
            self.norm = partial(nn.GroupNorm, num_groups=Utils.batch_norm_groups(ni))

        self.film = FilmBlock(n_emb, nf, act=act)
        self.conv1 = BasicBlocks.pre_conv(ni, nf, ks, act=act, norm=self.norm)
        self.conv2 = BasicBlocks.pre_conv(nf, nf, ks, act=act, norm=self.norm)
        self.idconv = fc.noop if ni == nf else nn.Conv2d(ni, nf, 1)
        self.attn = None
        self.cross_attn = None
        if attn_chans:
            self.attn = SelfAttentionPose(nf, n_pose, img_hw, attn_chans)
            self.cross_attn = CrossAttention(nf, attn_chans)

    def forward(self, x, t, pose_embed, cross_feat=None):
        inp = x
        x = self.conv1(x)
        x = self.film(x, t)
        x = self.conv2(x)
        x = x + self.idconv(inp)
        if self.attn is not None:
            x = x + self.attn(x, pose_embed)
            if cross_feat is not None:
                x = x + self.cross_attn(x, cross_feat)
        return x
