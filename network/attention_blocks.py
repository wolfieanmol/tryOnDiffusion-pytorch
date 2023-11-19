import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SelfAttentionPose(nn.Module):
    def __init__(self, ni, pose_dim, img_seq_dim, attn_chans, transpose=True):
        super().__init__()
        self.nheads = ni // attn_chans
        self.scale = math.sqrt(ni / self.nheads)
        self.norm = nn.LayerNorm(ni)
        self.x_pose_norm = nn.LayerNorm(ni + 2)
        self.pose_embed_layer = nn.Linear(pose_dim, img_seq_dim)
        self.q_layer = nn.Linear(ni, ni)
        self.k_layer = nn.Linear(ni + 2, ni)
        self.v_layer = nn.Linear(ni + 2, ni)
        self.proj = nn.Linear(ni, ni)
        self.transpose = transpose

    def forward(self, x, pose_embed):
        n, c, h, w = x.shape
        x = x.view(n, c, -1)  # n, c, s # pose_embed.shape = n, 2, pd
        pose_embed = self.pose_embed_layer(pose_embed)  # pose_embed.shape = n,2,s
        if self.transpose:
            x = x.transpose(1, 2)  # x.shape = n, s, c
            pose_embed = pose_embed.transpose(1, 2)  # n,s,1

        print(f"SelfAttentionPose - x.shape = {x.shape}, pose_embed.shape = {pose_embed.shape}")
        x_pose = torch.cat((x, pose_embed), dim=2)  # n, s, c+1
        x = self.norm(x)
        x_pose = self.x_pose_norm(x_pose)

        q, k, v = self.q_layer(x), self.k_layer(x_pose), self.v_layer(x_pose)  # n, s, c
        x = torch.cat((q, k, v), dim=-1)  # n,s,c
        x = rearrange(x, "n s (h d) -> (n h) s d", h=self.nheads)
        q, k, v = torch.chunk(x, 3, dim=-1)  # 16*n, s, c/16

        s = (q @ k.transpose(1, 2)) / self.scale
        x = s.softmax(dim=-1) @ v
        x = rearrange(x, "(n h) s d -> n s (h d)", h=self.nheads)
        x = self.proj(x)
        if self.transpose:
            x = x.transpose(1, 2)
        return x.reshape(n, c, h, w)


class CrossAttention(nn.Module):
    def __init__(self, ni, attn_chans, transpose=True):
        super().__init__()
        self.nheads = ni // attn_chans
        self.scale = math.sqrt(ni / self.nheads)
        self.q = nn.Sequential(nn.LayerNorm(ni), nn.Linear(ni, ni))
        self.kv = nn.Sequential(nn.LayerNorm(ni), nn.Linear(ni, ni * 2))
        self.proj = nn.Linear(ni, ni)
        self.transpose = transpose

    def forward(self, x, query_x):
        print(f"CrossAttention x = {x.shape}, query_x = {query_x.shape}")
        n, c, h, w = x.shape
        x = x.view(n, c, -1)
        query_x = query_x.view(n, c, -1)
        n, c, s = x.shape

        if self.transpose:
            x = x.transpose(1, 2)  # n,s,c
            query_x = query_x.transpose(1, 2)

        query_x = self.q(query_x)

        # repeat if there is single channel. Useful since this is also used as pool attention for clip embeddings.
        if not self.transpose and query_x.shape[1] == 1:
            query_x = query_x.repeat(1, 3, 1)

        x = self.kv(x)
        x = torch.cat((query_x, x), dim=-1)
        x = rearrange(x, "n s (h d) -> (n h) s d", h=self.nheads)
        q, k, v = torch.chunk(x, 3, dim=-1)
        s = (q @ k.transpose(1, 2)) / self.scale
        x = s.softmax(dim=-1) @ v
        x = rearrange(x, "(n h) s d -> n s (h d)", h=self.nheads)
        x = self.proj(x)

        if self.transpose:
            x = x.transpose(1, 2)
        return x.reshape(n, c, h, w)


class MultiHeadAttention(nn.Module):
    def __init__(self, ni, attn_chans):
        super().__init__()
        self.nheads = ni // attn_chans
        self.scale = math.sqrt(ni / self.nheads)
        self.q_proj = nn.Linear(ni, ni)
        self.k_proj = nn.Linear(ni, ni)
        self.v_proj = nn.Linear(ni, ni)
        self.c_proj = nn.Linear(ni, ni)
        # self.transpose = transpose

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.nheads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.permute(1, 0, 2)
