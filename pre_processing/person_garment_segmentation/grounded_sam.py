import os
import sys
from typing import Dict, List

# from pre_processing.person_garment_segmentation.dataloader import ModelImageDataset
from dataloader import ModelImageDataset
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

import argparse
import copy
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import requests
import supervision as sv
import torch
from huggingface_hub import hf_hub_download

# from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from segment_anything import SamPredictor, build_sam, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.ops import box_convert

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict, preprocess_caption
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


class Dino:
    def __init__(self, dino_config: Dict, batch_size: int, device: str = "cpu"):
        self.device = device
        self.batch_size = batch_size

        self.groundingdino_model = Dino._load_model_hf(**dino_config).to(self.device)

    def predict_batch(self, images_list, text_prompts, model, box_threshold=0.40, text_threshold=0.25):
        captions = [preprocess_caption(caption) for caption in text_prompts]

        images = torch.stack(images_list)
        images = images.to(self.device)

        with torch.no_grad():
            outputs = model(images, captions=captions)

        prediction_logits = outputs["pred_logits"].sigmoid()  # prediction_logits.shape = (bsz，nq, 256)
        prediction_boxes = outputs["pred_boxes"]  # prediction_boxes.shape = (bsz, nq, 4)

        logits_res = []
        boxs_res = []
        phrases_list = []
        tokenizer = model.tokenizer
        for ub_logits, ub_boxes, ub_captions in zip(prediction_logits, prediction_boxes, captions):
            mask = ub_logits.max(dim=1)[0] > box_threshold
            logits = ub_logits[mask]  # logits.shape = (n, 256)
            boxes = ub_boxes[mask]  # boxes.shape = (n, 4)
            logits_res.append(logits.max(dim=1)[0])
            boxs_res.append(boxes)

            tokenized = tokenizer(ub_captions)
            phrases = [
                get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace(".", "")
                for logit in logits
            ]
            phrases_list.append(phrases)
        return boxs_res, logits_res, phrases_list

    def detect(
        self,
        images: List,
        image_sources: List,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        print("DINO DETECT!!!!!!!!!!!!")

        batch_boxes, batch_logits, batch_phrases = self.predict_batch(
            images_list=images,
            text_prompts=[text_prompt] * self.batch_size,
            model=self.groundingdino_model,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # WHOLE PERSON
        batch_whole_person_indexes = [
            phrases.index("person") if "person" in phrases else None for phrases in batch_phrases
        ]
        batch_whole_person = [boxes[idx][None] for idx, boxes in zip(batch_whole_person_indexes, batch_boxes)]
        # batch_boxes = [torch.cat([box[None] for i, box in enumerate(boxes) if i != idx], dim=0) for idx, boxes in
        #                zip(batch_whole_person_indexes, batch_boxes)]

        batch_whole_person_boxes = []
        for image_source, boxes in zip(image_sources, batch_whole_person):
            H, W, _ = image_source.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H]).to(device)
            batch_whole_person_boxes.append(boxes_xyxy)

        # GARMENT
        batch_garment_indexes = [
            [i for i, x in enumerate(phrases) if x.replace(" ", "") == "t-shirt"] for phrases in batch_phrases
        ]

        return batch_boxes, batch_whole_person_boxes, batch_garment_indexes, batch_whole_person_indexes

    @staticmethod
    def _load_model_hf(repo_id, filename, ckpt_config_filename, device):
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

        args = SLConfig.fromfile(cache_config_file)
        args.device = device
        model = build_model(args)

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location=device)
        log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        return model


class Sam:
    def __init__(
        self,
        sam_checkpoint: str = "sam_hq_vit_h.pth",
        sam_model_type: str = "vit_h",
        batch_size: int = 2,
        device: str = "cpu",
    ):
        # "sam_vit_h_4b8939.pth"
        # "vit_h"
        self.device = device
        self.batch_size = batch_size

        self.sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).to(device=device)
        # self.sam_model = build_sam_hq(checkpoint=sam_checkpoint).to(device)
        self.sam_predictor = SamPredictor(self.sam_model)

    def segment(self, batched_input, garment_indexs):
        batched_output = self.sam_model(batched_input, multimask_output=False)

        batch_garment_masks, batch_masks = [], []
        for output, garment_index in zip(batched_output, garment_indexs):
            masks = output["masks"]
            garment_mask = masks[garment_index]
            masks = torch.cat([masks[:garment_index], masks[garment_index + 1 :]], dim=0)
            batch_garment_masks.append(garment_mask.cpu()), batch_masks.append(masks.cpu())

        return batch_garment_masks, batch_masks

    def segment2(self, image_sources, batch_boxes: List, batch_garment_indexes: List, batch_whole_person_indexes: List):
        print("SAM SEGMENT!!!!!!!!!!!!")
        batch_garment_masks, batch_person_masks = [], []
        for image_source, boxes, garment_indexes, whole_person_index in zip(
            image_sources, batch_boxes, batch_garment_indexes, batch_whole_person_indexes
        ):
            self.sam_predictor.set_image(image_source)
            H, W, _ = image_source.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H]).to(device)

            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
                boxes_xyxy.to(self.device), image_source.shape[:2]
            )
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            garment_mask = torch.zeros(1, *masks.shape[1:])
            if len(garment_indexes) > 0:
                garment_mask = masks[garment_indexes[0]][None]
                garment_indexes.append(whole_person_index)
                masks = torch.cat([mask[None] for i, mask in enumerate(masks) if i not in garment_indexes], dim=0)

            batch_garment_masks.append(garment_mask.cpu())
            batch_person_masks.append(masks.cpu())

        return batch_garment_masks, batch_person_masks


class Segment:
    def __init__(
        self,
        dino_config: Dict,
        sam_checkpoint: str = "sam_hq_vit_h.pth",
        sam_model_type: str = "vit_h",
        batch_size: int = 2,
        device: str = "cpu",
        root_dir: str = "",
    ):
        self.device = device
        self.batch_size = batch_size
        self.root_dir = root_dir

        # self.image_sources, self.images = list(zip(*map(load_image, self.inputs["image_names"])))

        self.dino = Dino(dino_config, batch_size=batch_size, device=device)
        self.sam = Sam(sam_checkpoint, sam_model_type, batch_size=batch_size, device=device)

    def process(self, inputs: Dict, text_prompt: str, box_threshold: float = 0.35, text_threshold: float = 0.25):
        self.inputs = inputs
        image_sources, images = list(zip(*map(load_image, inputs["image_paths"])))

        batch_boxes, batch_whole_person_boxes, batch_garment_indexes, batch_whole_person_indexes = self.dino.detect(
            images=images,
            image_sources=image_sources,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        batch_garment_masks, batch_person_masks = self.sam.segment2(
            image_sources, batch_boxes, batch_garment_indexes, batch_whole_person_indexes
        )

        return image_sources, batch_garment_masks, batch_person_masks, batch_whole_person_boxes

    def convert_box(self, image, detected_boxes):
        H, W, _ = image.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(detected_boxes) * torch.Tensor([W, H, W, H]).to(device)
        transformed_boxes = self.sam.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image.shape[:2])
        return transformed_boxes

    def create_images(self, image_sources, batch_garment_masks, batch_person_masks, batch_whole_person_boxes):
        for image_path, image_name, garment_mask, person_masks, person_box in zip(
            image_sources, self.inputs["image_names"], batch_garment_masks, batch_person_masks, batch_whole_person_boxes
        ):
            combined_mask = torch.zeros(1, *person_masks.shape[1:])
            for mask in person_masks:
                combined_mask += mask[None]
            combined_mask[combined_mask != 0] = 1
            garment_mask[garment_mask != 0] = 1

            self.save_image(image_path, combined_mask, image_name, "person", "cloth_agnostic", person_box)
            self.save_image(image_path, garment_mask, image_name, "garment", "cloth")

    def save_image(self, image_path, mask, image_name, mask_name, category, person_box=None):
        image = Image.open(image_path)
        masked_image_array = np.array(image) * (mask.permute(2, 3, 0, 1).squeeze(-1).numpy()).astype(np.uint8)

        if person_box is not None:
            masked_image = Image.fromarray(np.uint8(masked_image_array)).convert("RGBA")
            draw = ImageDraw.Draw(image)
            draw.rectangle(person_box.tolist()[0], fill=(128, 128, 128))
            # rect_array = np.array(image)

            image2 = image.convert("RGBA")

            masked_image = np.array(masked_image)
            image2 = np.array(image2)

            new_image = np.where(masked_image > 0, masked_image, image2)
            im = Image.fromarray(new_image.astype(np.uint8)).convert("RGB")
        else:
            im = Image.fromarray(np.uint8(masked_image_array))

        filename = f"{self.root_dir}/{category}/{image_name}_{mask_name}.jpg"
        print(f"saving image {filename}")
        im.save(filename)

    def _prepare_image(self, image, transform):
        image = transform.apply_image(image)
        image = torch.as_tensor(image, device=self.device)
        return image.permute(2, 0, 1).contiguous()


if __name__ == "__main__":
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "/content/data/dataset/test/image/test"
    test = ModelImageDataset(data_path)
    dataloader = DataLoader(test, batch_size=batch_size, num_workers=2)

    dino_config = {
        "repo_id": "ShilongLiu/GroundingDINO",
        "filename": "groundingdino_swinb_cogcoor.pth",
        "ckpt_config_filename": "GroundingDINO_SwinB.cfg.py",
        "device": device,
    }

    root_dir = "/content/drive/MyDrive/datasets/viton-agnostic/dinosam"
    segment = Segment(dino_config, batch_size=batch_size, device=device, root_dir=root_dir)

    for batch_names, bath_paths, batch_inputs in dataloader:
        inputs = {
            "image_names": batch_names,
            "image_paths": bath_paths,
            "raw_images": batch_inputs,
        }

        prompt = "hands, face with hair, t-shirt, legs, person"
        image_sources, batch_garment_masks, batch_person_masks, batch_whole_person_boxes = segment.process(
            inputs, text_prompt=prompt
        )
        segment.create_images(inputs["image_paths"], batch_garment_masks, batch_person_masks, batch_whole_person_boxes)
