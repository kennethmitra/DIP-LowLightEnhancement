from ImageDataset import ImageDataset
from Model import EnhancerModel
from pathlib import Path
from LossFunctions import *
import re
import matplotlib.pyplot as plt
from PIL import Image


def enhance(image_name=None):
    DATA_DIR = "static/uploads/"
    SAVE_DIR = "static/enhanced/"
    test_dataset = ImageDataset(DATA_DIR, 1)
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)


    # Get compute device
    print("-------------------------------GPU INFO--------------------------------------------")
    print('Available devices ', torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current cuda device ', device)
    if device != "cpu":
        print('Current CUDA device name ', torch.cuda.get_device_name(device))
    print("-----------------------------------------------------------------------------------")

    # Load Saved Model
    save_file = "./saves/j3/epo2.save"
    model_state = torch.load(save_file)['model_state']

    model = EnhancerModel().to(device)
    model.load_state_dict(model_state)

    image = Image.open(DATA_DIR + image_name)

    with torch.no_grad():
        image = image.unsqueeze(dim=0).to(device)  # Add batch dimension and send to GPU
        curves = model(image)
        enhanced_image = model.enhance_image(image, curves)
        curves = model(image)
        enhanced_image = model.enhance_image(image, curves)

        curves = torch.stack(torch.split(curves, split_size_or_sections=3, dim=1), dim=1)
        image = image.squeeze().permute(1, 2, 0).cpu().flip(dims=[0, 1])
        curves = (curves.squeeze().permute(0, 2, 3, 1).mean(dim=0).cpu()) / 2 + 0.5
        enhanced_image2 = enhanced_image.squeeze().permute(1, 2, 0).cpu()
        enhanced_image2 = enhanced_image2.cpu().detach().numpy()
        # plt.imshow(enhanced_image2)
        # plt.show()
        plt.imsave(SAVE_DIR + image_name + ".jpg", enhanced_image2)
