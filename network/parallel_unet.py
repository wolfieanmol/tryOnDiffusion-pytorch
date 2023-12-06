import torch
import torch.nn as nn

from network.unet import EmbUNetModel


class PersonUnetModel(EmbUNetModel):
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
        super().__init__(
            pose_embed_dim, in_channels, out_channels, img_res, attn_chans, nfs, num_layers, attn_layers, up_blocks
        )


class GarmentUnetModel(EmbUNetModel):
    def __init__(
        self,
        pose_embed_dim,
        in_channels=3,
        out_channels=3,
        img_res=128,
        attn_chans=0,
        nfs=(128, 256, 512, 1024),
        num_layers=(3, 4, 6, 7),
        attn_layers=(False, False, False, False),
        up_blocks=(True, True, False, False),
    ):
        super().__init__(
            pose_embed_dim, in_channels, out_channels, img_res, attn_chans, nfs, num_layers, attn_layers, up_blocks
        )


class ParallelUnetModel128(nn.Module):
    def __init__(self, nfs=(128, 256, 512, 1024), img_res=128, attn_chans=8, **kwargs):
        super().__init__()
        self.garment_unet = GarmentUnetModel(pose_embed_dim=16, nfs=nfs, img_res=img_res, attn_chans=attn_chans)
        self.person_unet = PersonUnetModel(pose_embed_dim=16, nfs=nfs, img_res=img_res, attn_chans=attn_chans)

    def forward(self, inp):
        # device = "cuda"
        # person_image = torch.rand(2, 6, 128, 128).to(device)
        # garment_image = torch.rand(2, 3, 128, 128).to(device)
        # pose_embed = torch.rand(2, 2, 16).to(device)
        # t = torch.rand(2).to(device)
        # t_na = torch.rand(2).to(device)

        garment_image, person_image, pose_embed, t, t_na = inp
        inp_garment = (garment_image, pose_embed, None, t_na)
        inp_person = (person_image, pose_embed, t, t_na)

        garment_out = self.garment_unet(inp_garment)
        print(f"$$$$$ {type(garment_out)}")

        garment_down_block_features = [p for o in self.garment_unet.downs for p in o.saved_block_features]
        print(f"22222$$$$$ {type(garment_down_block_features)}")
        garment_up_block_features = [p for o in self.garment_unet.ups for p in o.saved_block_features]
        print(f"22222$$$$$ {type(garment_out)}")
        out = self.person_unet(inp_person, (garment_down_block_features, garment_up_block_features))
        return out
