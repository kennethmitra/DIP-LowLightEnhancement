from fastiqa.all import *
from PIL import Image
import torch


data = Im2MOS(TestImages)
model = RoIPoolModel()

learn = RoIPoolLearner(data, model, path='.')
learn.load('RoIPoolModel')

image1 = Image.open('images/train_data/1.jpg')
image1 = image1.resize((256, 256))               # Crop to square
image1 = torch.from_numpy(np.asarray(image1).astype(float)).float() / 255.0   # Convert to tensor
image1 = image1.permute(2, 0, 1)

image2 = Image.open('images/train_data/2.jpg')
image2 = image2.resize((256, 256))               # Crop to square
image2 = torch.from_numpy(np.asarray(image2).astype(float)).float() / 255.0   # Convert to tensor
image2 = image2.permute(2, 0, 1)

image3 = Image.open('images/train_data/230.jpg')
image3 = image3.resize((256, 256))               # Crop to square
image3 = torch.from_numpy(np.asarray(image3).astype(float)).float() / 255.0   # Convert to tensor
image3 = image3.permute(2, 0, 1)

im = torch.stack([image1, image2, image3])

for i in range(im.shape[0]):
    qmap = learn.predict_quality_map(im[i, :, :, :], [5, 5])
    print(qmap.global_score)

