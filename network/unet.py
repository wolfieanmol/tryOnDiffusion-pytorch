import torch.nn as nn

from network.basic_blocks import BasicBlocks
from network.embeddings import ConditionalEmbedding
from network.unet_blocks import DownBlock, UpBlock


class EmbUNetModel(nn.Module):
    def __init__(
        self,
        pose_embed_dim,
        in_channels=6,
        out_channels=3,
        img_res=128,
        attn_chans=8,
        nfs=(128, 256, 512, 1024),
        num_layers=(3, 4, 6, 7),
        attn_layers=(False, False, True, True),
        up_blocks=(True, True, True, True),
    ):
        super().__init__()
        print(nfs)
        self.up_blocks = up_blocks

        self.conv_in = nn.Conv2d(in_channels, nfs[0], kernel_size=3, padding=1)

        self.n_temb = nf = nfs[0]
        n_emb = nf * 8
        self.conditional_embedding = ConditionalEmbedding(self.n_temb, pose_embed_dim, attn_chans=8)
        self.emb_mlp = nn.Sequential(
            BasicBlocks.lin(self.n_temb, n_emb, norm=nn.BatchNorm1d), BasicBlocks.lin(n_emb, n_emb)
        )

        self.downs = nn.ModuleList()
        n = len(nfs)
        for i in range(n):
            ni = nf
            nf = nfs[i]
            self.downs.append(
                DownBlock(
                    n_emb,
                    pose_embed_dim,
                    ni,
                    nf,
                    img_res ** 2,
                    add_down=i != n - 1,
                    num_layers=num_layers[i],
                    attn_chans=attn_chans if attn_layers[i] else 0,
                )
            )
            if i != n - 1:
                img_res = img_res // 2
        # self.mid_block = EmbResBlock(n_emb, pose_embed_dim, nfs[-1], nfs[-1], img_res)

        rev_nfs = list(reversed(nfs))
        nf = rev_nfs[0]
        self.ups = nn.ModuleList()
        for i in range(n):
            prev_nf = nf
            nf = rev_nfs[i]
            ni = rev_nfs[min(i + 1, len(nfs) - 1)]
            if up_blocks[i]:
                self.ups.append(
                    UpBlock(
                        n_emb,
                        pose_embed_dim,
                        ni,
                        prev_nf,
                        nf,
                        img_res ** 2,
                        add_up=i != n - 1,
                        num_layers=num_layers[n - i - 1] + 1,
                        attn_chans=attn_chans if attn_layers[i] else 0,
                    )
                )
                if i != n - 1:
                    img_res = img_res * 2
        self.conv_out = BasicBlocks.pre_conv(nfs[0], out_channels, act=nn.SiLU, bias=False)

    def forward(self, inp, cross_attn_features=None):
        x, pose_embed, t, t_na = inp
        down_feats, up_feats = cross_attn_features if cross_attn_features is not None else ([], [])

        temb = self.conditional_embedding(pose_embed, t, t_na)
        print(temb.shape)  # 2, 3, 128
        emb = self.emb_mlp(temb)

        x = self.conv_in(x)
        saved = [x]
        for block in self.downs:
            x = block(x, emb, pose_embed, down_feats)
            print(f"DOWN_BLOCK -- {x.shape}")
        saved += [p for o in self.downs for p in o.saved]
        # x = self.mid_block(x, emb, pose_embed)
        print(f"MID_BLOCK -- {x.shape}")
        for block in self.ups:
            x = block(x, emb, pose_embed, saved, up_feats)
            print(f"UP_BLOCK -- {x.shape}")

        if self.up_blocks[-1]:
            x = self.conv_out(x)
        print("############# END U-NET #############")
        print(f"$$$$$ {type(x)}")
        return x
