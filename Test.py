import torch
from ImageDataset import ImageDataset
from Model import EnhancerModel
from torchvision.utils import save_image
from pathlib import Path

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
save_file = "./saves/test1/epo10.save"
model_state = torch.load(save_file)['model_state']

model = EnhancerModel().to(device)
model.load_state_dict(model_state)

with torch.no_grad():
    for img_num, image in enumerate(test_dataset):
        image = image.unsqueeze(dim=0).to(device)  # Add batch dimension and send to GPU
        curves = model(image)
        enhanced_image = model.enhance_image(image, curves)

        enhanced_image = enhanced_image.flip(dims=[2, 3])
        save_image(enhanced_image, f"{SAVE_DIR}/{Path(test_dataset.image_names[img_num]).stem}_enhanced.jpg")

print("DONE")

