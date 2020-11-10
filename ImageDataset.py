from torch.utils.data import Dataset
import torch
from PIL import Image
import glob
import numpy as np

class ImageDataset(Dataset):
    
    def __init__(self, image_dir, image_dim, image_list=None):
        """
        Image Dataset for Low Light Enhancer
        """
        self.image_dir = image_dir
        self.image_dim = image_dim

        if not image_list:
            self.image_names = glob.glob(f"{self.image_dir}/*.jpg")
        else:
            self.image_names = [f"{self.image_dir}/{x}" for x in image_list]

    
    def __getitem__(self, index):
        image = Image.open(self.image_names[index])

        if self.image_dim > 0:
            image.thumbnail((self.image_dim, self.image_dim), Image.ANTIALIAS)
            if image.width != self.image_dim or image.height != self.image_dim:
                print(f"WARNING: Image dimension ({image.width}, {image.height}) is different from specified ({self.image_dim})")

        image = torch.from_numpy(np.asarray(image).astype(float)).float() / 255.0   # Convert to tensor
        image = image.permute(2, 0, 1)                                              # Convert to (n,c,h,w) order
        return image

    def __len__(self):
        return len(self.image_names)

if __name__ == '__main__':
    ds = ImageDataset("./images/test_data", -1)