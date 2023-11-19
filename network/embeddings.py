import torch
import torch.nn as nn
import torch.nn.functional as F

from network.attention_blocks import MultiHeadAttention
from network.basic_blocks import BasicBlocks
from utils.utils import GaussianSmoothing


class ConditionalEmbedding(nn.Module):
    def __init__(self, t_embed_dim, pose_embed_dim, attn_chans, noise_level=0.3):
        super().__init__()
        self.pose_embed_dim = pose_embed_dim
        self.t_embed_dim = t_embed_dim
        self.positional_embedding = nn.Parameter(torch.randn(3, pose_embed_dim) / pose_embed_dim ** 0.5)
        self.lin = nn.Sequential(nn.LayerNorm(pose_embed_dim), nn.Linear(pose_embed_dim, t_embed_dim))
        self.pool_attn = MultiHeadAttention(t_embed_dim, attn_chans)
        self.smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=noise_level, conv_dim=1)

    def forward(self, x, t, t_na):
        # x.shape = NCP
        x = torch.cat((x.mean(dim=1, keepdim=True), x), dim=1)  # N,C+1,P
        x += self.positional_embedding[None, :, :].to(x.dtype)
        x = self.lin(x)  # N,C+1,T
        x = self.pool_attn(x)  # N,1,T

        x_noisy = self.smoothing(F.pad(x, (1, 1), mode="reflect"))
        x = x + x_noisy

        # TODO should x be normalized before/after adding timestep embeddings for proper scaling?
        if t is not None:
            x += BasicBlocks.timestep_embedding(t, self.t_embed_dim)[:, None, :]
        x += BasicBlocks.timestep_embedding(t_na, self.t_embed_dim)[:, None, :]
        return x.squeeze(1)
