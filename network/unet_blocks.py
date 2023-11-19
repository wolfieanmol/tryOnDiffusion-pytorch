import torch
import torch.nn as nn

from network.basic_blocks import BasicBlocks
from network.resblock import ResBlock
from utils.utils import Utils


class UnetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.saved = []
        self.saved_block_features = []


class DownBlock(UnetBlock):
    def __init__(
        self,
        n_emb,
        n_pose,
        ni,
        nf,
        img_hw=None,
        add_down=True,
        num_layers=1,
        attn_chans=0,
        save_skip_feat=True,
        save_cross_feat=True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                Utils.saved(
                    ResBlock(n_emb, n_pose, ni if i == 0 else nf, nf, img_hw, attn_chans=attn_chans),
                    self,
                    save_skip_feat,
                    save_cross_feat,
                )
                for i in range(num_layers)
            ]
        )
        self.down = (
            Utils.saved(nn.Conv2d(nf, nf, 3, stride=2, padding=1), self, save_skip_feat, save_cross_feat=False)
            if add_down
            else nn.Identity()
        )

    def forward(
        self, x=torch.rand(2, 6, 64, 64), t=torch.rand(2), pose_embed=torch.rand(2, 2, 16), cross_features=None
    ):
        for resnet in self.resnets:
            cross_feat = cross_features.pop(0) if cross_features else None
            x = resnet(x, t, pose_embed, cross_feat)

        x = self.down(x)
        return x


class UpBlock(UnetBlock):
    def __init__(
        self,
        n_emb,
        n_pose,
        ni,
        prev_nf,
        nf,
        img_hw=None,
        add_up=True,
        num_layers=2,
        attn_chans=0,
        save_skip_feat=False,
        save_cross_feat=True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                Utils.saved(
                    ResBlock(
                        n_emb,
                        n_pose,
                        (prev_nf if i == 0 else nf) + (ni if (i == num_layers - 1) else nf),
                        nf,
                        img_hw,
                        attn_chans=attn_chans,
                    ),
                    self,
                    save_skip_feat,
                    save_cross_feat,
                )
                for i in range(num_layers)
            ]
        )
        self.up = BasicBlocks.upsample(nf) if add_up else nn.Identity()

    def forward(self, x, t, pose_embed, ups, cross_features=None):
        for resnet in self.resnets:
            cross_feat = cross_features.pop(0) if cross_features else None
            x = resnet(torch.cat([x, ups.pop()], dim=1), t, pose_embed, cross_feat)
        return self.up(x)
