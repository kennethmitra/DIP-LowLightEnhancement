import torch
from ImageDataset import ImageDataset
from Model import EnhancerModel
from torchvision.utils import save_image
from pathlib import Path
import matplotlib.pyplot as plt
from LossFunctions import *

DATA_DIR = "images/progress_pics"
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
save_file = "./saves/col_var_loss_enabled3/epo1.save"
model_state = torch.load(save_file)['model_state']

model = EnhancerModel().to(device)
model.load_state_dict(model_state)

#fig, ax = plt.subplots(len(test_dataset), 4, figsize=(17, 10))

with torch.no_grad():
    for img_num, image in enumerate(test_dataset):
        image = image.unsqueeze(dim=0).to(device)  # Add batch dimension and send to GPU
        #curves = model(image)
        #enhanced_image = model.enhance_image(image, curves)
#
#         curves = torch.stack(torch.split(curves, split_size_or_sections=3, dim=1), dim=1)
#
#         image = image.squeeze().permute(1, 2, 0).cpu().flip(dims=[0, 1])
#         curves = (curves.squeeze().permute(0, 2, 3, 1).mean(dim=0).cpu())/2+0.5
#         enhanced_image = enhanced_image.squeeze().permute(1, 2, 0).cpu()
#
#
#         ax[img_num][0].imshow(image)
#         ax[img_num][0].set_title("Original")
#         ax[img_num][1].imshow(curves)
#         ax[img_num][1].set_title("RGB Curves")
#         ax[img_num][2].imshow(curves.mean(dim=2))
#         ax[img_num][2].set_title("Grayscale Curves")
#         ax[img_num][3].imshow(enhanced_image)
#         ax[img_num][3].set_title("Enhanced Image")
#
#         enhanced_image = enhanced_image.flip(dims=[0, 1])
#         plt.show()
        # save_image(enhanced_image, f"{OUTPUT_DIR}/{Path(test_dataset.image_names[img_num]).stem}_enhanced.jpg")
        exposure_loss = ExposureControlLoss(gray_value=0.4, patch_size=16, method=1, device=device)  # Using method 2 based on bsun0802's code
        exp_orig = exposure_loss(image)
        iq_loss = ImageQualityLoss(method=2, device=device,
                                   blk_size=(3, 5))  # From https://github.com/baidut/paq2piq/blob/master/demo.ipynb

        fig, ax = plt.subplots(4, 4, figsize=(17, 10))
        for i in range(16):
            if i > 0:
                curves = model(image)
                enhanced_image = model.enhance_image(image, curves)
            else:
                enhanced_image = image
            iq_loss_val = torch.mean(iq_loss(enhanced=enhanced_image, original=image)).item()
            exp_loss_val = torch.mean(exposure_loss(image)).item()
            image = image.squeeze().permute(1, 2, 0).cpu().flip(dims=[0, 1])
            enhanced_image2 = enhanced_image.squeeze().permute(1, 2, 0).cpu()
            ax[i//4][i%4].imshow(enhanced_image2)

            iq_total_loss = ((1 + -iq_loss_val)**2 - 1)
            exp_total_loss = ((1 + exp_orig - exp_loss_val)**2 - 1) * 1.5
            ax[i//4][i%4].set_title("Loss = " + str(iq_total_loss + exp_total_loss))
            image = enhanced_image
            print(f'{iq_total_loss} {exp_total_loss}')
        plt.show()




print("DONE")

