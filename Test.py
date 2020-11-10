import torch
from ImageDataset import ImageDataset
from Model import EnhancerModel
from torchvision.utils import save_image
from pathlib import Path
import matplotlib.pyplot as plt
from LossFunctions import *

DATA_DIR = "images/test_data"
SAVE_DIR = "images/test_data"
test_dataset = ImageDataset(DATA_DIR, 512)
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

print(f"Number of images in test: {len(test_dataset)}")

# Get compute device
print("-------------------------------GPU INFO--------------------------------------------")
print('Available devices ', torch.cuda.device_count())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current cuda device ', device)
if device != "cpu":
    print('Current CUDA device name ', torch.cuda.get_device_name(device))
print("-----------------------------------------------------------------------------------")

# Load Saved Model
save_file = "./saves/orig_code/epo41.save"
model_state = torch.load(save_file)['model_state']

model = EnhancerModel().to(device)
model.load_state_dict(model_state)

ill_loss = IlluminationSmoothnessLoss(method=1)

fig, ax = plt.subplots(len(test_dataset), 4, figsize=(17, 10))

with torch.no_grad():
    for img_num, image in enumerate(test_dataset):
        image = image.unsqueeze(dim=0).to(device)  # Add batch dimension and send to GPU
        curves = model(image)
        ill_loss_val = ill_loss(curves)
        enhanced_image = model.enhance_image(image, curves)

        curves = torch.stack(torch.split(curves, split_size_or_sections=3, dim=1), dim=1)

        image = image.squeeze().permute(1, 2, 0).cpu().flip(dims=[0, 1])
        curves = (curves.squeeze().permute(0, 2, 3, 1).mean(dim=0).cpu()).flip(dims=[0, 1])/2+0.5
        enhanced_image = enhanced_image.squeeze().permute(1, 2, 0).cpu().flip(dims=[0, 1])


        ax[img_num][0].imshow(image)
        ax[img_num][0].set_title("Original")
        ax[img_num][1].imshow(curves)
        ax[img_num][1].set_title("RGB Curves")
        ax[img_num][2].imshow(curves.mean(dim=2))
        ax[img_num][2].set_title("Grayscale Curves")
        ax[img_num][3].imshow(enhanced_image)
        ax[img_num][3].set_title("Enhanced Image")

        print(f"Ill_Loss: {ill_loss_val}")

        enhanced_image = enhanced_image.flip(dims=[0, 1])
        # save_image(enhanced_image, f"{SAVE_DIR}/{Path(test_dataset.image_names[img_num]).stem}_enhanced.jpg")

plt.show()

print("DONE")
