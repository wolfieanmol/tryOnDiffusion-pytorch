import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ModelImageDataset(Dataset):
    def __init__(self, image_dir):
        print(os.getcwd())
        print(image_dir)
        self.model_image_dir = image_dir
        self.model_image_files = os.listdir(self.model_image_dir)
        self.num_images = len(self.model_image_files)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        model_image_file = self.model_image_files[index % self.num_images]
        model_image_path = os.path.join(self.model_image_dir, model_image_file)

        with Image.open(model_image_path) as im:
            im.save(model_image_path)
            model_image = np.array(im.convert("RGB"))

        return model_image_file, model_image_path, model_image


if __name__ == "__main__":
    data_path = "/content/data/dataset/test/image"
    test = ModelImageDataset(data_path)
    dataloader = DataLoader(test, batch_size=4, num_workers=2)
    for file_name, file_path, img in dataloader:
        print(file_name, file_path)
        break
