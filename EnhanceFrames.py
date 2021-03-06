import os

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from ImageDataset import ImageDataset
from Model import EnhancerModel
from pathlib import Path
from LossFunctions import *
from torchvision.utils import save_image
from collections import defaultdict
import cv2
import re


from Deep_White_Balance.arch import deep_wb_model
import Deep_White_Balance.utilities.utils as utls
from Deep_White_Balance.utilities.deepWB import deep_wb
import Deep_White_Balance.arch.splitNetworks as splitter
from Deep_White_Balance.arch import deep_wb_single_task


def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

class ImageQualitySelector:
    def __init__(self):
        self.exposure_loss = ExposureControlLoss(gray_value=0.4, patch_size=16, method=1,
                                            device=device)  # Using method 2 based on bsun0802's code
        self.iq_loss = ImageQualityLoss(method=2, device=device, blk_size=(3, 5))  # From https://github.com/baidut/paq2piq/blob/master/demo.ipynb

    def select_best(self, original_image, enhanced_images):
        with torch.no_grad():
            enhanced_scores = []
            for enhanced_image in enhanced_images:
                iq_loss_val = torch.mean(self.iq_loss(enhanced=enhanced_image.unsqueeze(0), original=original_image.unsqueeze(0))).item()
                exp_loss_val = torch.mean(self.exposure_loss(original_image.unsqueeze(0))).item() - torch.mean(self.exposure_loss(original_image.unsqueeze(0))).item()
                iq_total_loss = ((1 + -iq_loss_val) ** 2 - 1)
                exp_total_loss = ((1 + exp_loss_val) ** 2 - 1) * 1.5
                #print(f'{iq_total_loss} {exp_total_loss}')
                enhanced_scores.append(iq_total_loss + exp_total_loss)
                # plt.imshow(transforms.ToPILImage()(enhanced_image))
                # plt.title(enhanced_scores[-1])
                # plt.show()
            max_index, max_score = max(enumerate(enhanced_scores), key=lambda x: x[1])
            #print(f"Min val: {max_score}")
        return enhanced_images[max_index], max_index, max_score





# Get compute device
print("-------------------------------GPU INFO--------------------------------------------")
print('Available devices ', torch.cuda.device_count())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current cuda device ', device)
if device != "cpu":
    print('Current CUDA device name ', torch.cuda.get_device_name(device))
print("-----------------------------------------------------------------------------------")


INPUT_DIR = "images/videos/video1"
OUTPUT_DIR = "images/video1_output/"
# INPUT_DIR = "images/test_data"
# OUTPUT_DIR = "images/test_output"
FILE_EXTENSION = ".png"

assert Path(INPUT_DIR).exists()
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# List of Models to use (Folder name, save file location, skip_predictions)
# We don't run predictions on frames that already exist but checking if the intermed_files exist takes time
# so we have another level of skipping in the models tuple

save_files = (("J2_epo1", "./models/j2_epo1.save", False),
              ("J2_epo3", "./models/j2_epo3.save", False),
              ("J3_epo2", "./models/j3_epo2.save", False),
              ("J4_epo3", "./models/j4_epo3.save", False),
              ("Uhoh_epo200", "./models/uhoh_test_gamma_A_2_epo200.save", False),
              ("deexp2_epo2", "./models/deexp2_epo1.save", False))

STAGE_SELECTORS = (False, False, False, True)

# Create dataset
test_dataset = ImageDataset(INPUT_DIR, 512, f_ext=FILE_EXTENSION, sort_key=natural_keys, suppress_warnings=True)
len_test_dataset = len(test_dataset)
print(f"Number of images in test: {len_test_dataset}")


############################################################
#                       Enhance Images                     #
############################################################
if STAGE_SELECTORS[0]:
    # We can't fit all the models in memory at once, so we run them one at a time

    for save_file in save_files:

        # Skip model if skip_predictions is True
        if save_file[2]:
            continue

        cwd = f"{OUTPUT_DIR}/{save_file[0]}"
        Path(cwd).mkdir(parents=True, exist_ok=True)

        # Load model
        model = EnhancerModel().to(device)
        saved_info = torch.load(save_file[1])
        model_state = saved_info['model_state']
        model.load_state_dict(model_state)
        print(f"Loading model {save_file[0]} from path {save_file[1]}, snapshot of epoch {saved_info['epoch']}")

        with torch.no_grad():
            for img_num, image in enumerate(test_dataset):
                image_name = f"{cwd}/frame_{img_num}{FILE_EXTENSION}"

                if not Path(image_name).exists():
                    image = image.unsqueeze(dim=0).to(device)  # Add batch dimension and send to GPU
                    curves = model(image)                      # Predict curves with model
                    enhanced_image = model.enhance_image(image, curves)  # Apply curves to image

                    save_image(enhanced_image, image_name)
                else:
                    print(f"Skipping {image_name} because it already exists")

                try:
                    if img_num % (len_test_dataset // 20) == 0:
                        print(f"\t{img_num / len_test_dataset * 100 :.2f}% complete")
                except:
                    pass


############################################################
#                   Select Enhanced Images                 #
############################################################

if STAGE_SELECTORS[1]:
    print("Selecting best images...")

    imageSelector = ImageQualitySelector()

    # Create datasets for each model's enhanced images
    image_datasets = {}
    for save_file in save_files:

        cwd = f"{OUTPUT_DIR}/{save_file[0]}"
        if not Path(cwd).exists():
            continue

        image_datasets[save_file[0]] = ImageDataset(cwd, 512, f_ext=FILE_EXTENSION, sort_key=natural_keys, suppress_warnings=True)

    # Make sure all datasets have all frames of video
    length = -1
    for ds_name in image_datasets:
        if length >= 0:
            assert len(image_datasets[ds_name]) == length
        else:
            length = len(image_datasets[ds_name])

    cwd = f"{OUTPUT_DIR}/selected"
    Path(cwd).mkdir(parents=True, exist_ok=True)

    # Step through datasets simultaneously and chose best images
    scores_hist = []
    model_freq = defaultdict(lambda: 0)
    for img_num, images in enumerate(zip(test_dataset, *image_datasets.values())):
        selectedImage, max_idx, max_score = imageSelector.select_best(original_image=images[0], enhanced_images=images)
        save_image(selectedImage, f'{cwd}/frame_{img_num}{FILE_EXTENSION}')
        scores_hist.append(max_score)

        if max_idx != 0:
            model_freq[save_files[max_idx-1][0]] += 1
        else:
            model_freq['Original'] += 1

        try:
            if img_num % (len_test_dataset // 20) == 0:
                 print(f"\t{img_num / len_test_dataset * 100 :.2f}% complete")
        except:
            pass

    # Print out which models were used how frequently
    print("Model popularity")
    for model_name, freq in model_freq.items():
        print(f"{model_name}\t{freq}")

    plt.plot(scores_hist)
    plt.show()

############################################################
#                     Color Grading                        #
############################################################
if STAGE_SELECTORS[2]:

    print("Correcting White Balance...")

    net_awb = deep_wb_single_task.deepWBnet()
    net_awb.to(device=device)
    net_awb.load_state_dict(torch.load(os.path.join("./Deep_White_Balance/models", 'net_awb.pth'), map_location=device))
    net_awb.eval()

    # Make sure input directory exists
    assert Path(f"{OUTPUT_DIR}/selected").exists()

    # Create and set output dir
    cwd = f"{OUTPUT_DIR}/awb"
    Path(cwd).mkdir(parents=True, exist_ok=True)

    # Auto White Balance
    file_names = glob.glob(f"{OUTPUT_DIR}/selected/*{FILE_EXTENSION}")
    file_names.sort(key=natural_keys)

    for i, image_name in enumerate(file_names):
        image = Image.open(image_name)
        out_awb = deep_wb(image, task='awb', net_awb=net_awb, device=device, s=656)
        result_awb = utls.to_image(out_awb)
        result_awb.save(f"{cwd}/frame_awb_{i}{FILE_EXTENSION}")

        try:
            if i % (len_test_dataset // 20) == 0:
                 print(f"\t{i / len_test_dataset * 100 :.2f}% complete")
        except:
            pass

############################################################
#             Adaptive Histogram Equalization              #
############################################################


if STAGE_SELECTORS[3]:
    print("Applying Contrast Limited Adaptive Histogram Equalization...")

    # Make sure input directory exists
    input_dir = f"{OUTPUT_DIR}/selected"
    assert Path(input_dir).exists()

    # Create and set output dir
    cwd = f"{OUTPUT_DIR}/clahe"
    Path(cwd).mkdir(parents=True, exist_ok=True)

    # Get list of file names
    file_names = glob.glob(f"{input_dir}/*{FILE_EXTENSION}")
    file_names.sort(key=natural_keys)

    # Apply Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    for i, image_name in enumerate(file_names):
        image = cv2.imread(image_name)
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_lab[..., 2] = clahe.apply(image_lab[..., 2])
        image = cv2.cvtColor(image_lab, cv2.COLOR_HSV2RGB)
        result_clahe = Image.fromarray(image)
        result_clahe.save(f"{cwd}/frame_he_{i}{FILE_EXTENSION}")

        try:
            if i % (len_test_dataset // 20) == 0:
                 print(f"\t{i / len_test_dataset * 100 :.2f}% complete")
        except:
            pass