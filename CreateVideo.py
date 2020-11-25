import cv2
import numpy as np
import os
from os.path import isfile, join
import re
from PIL import Image
import glob

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]


pathIn_original = './images/videos/video1'
pathIn_intermed = './images/video1_output/selected'
pathIn_output = './images/video1_output/clahe'
pathOut = 'enhanced6.avi'

fps = 25

original_files = glob.glob(f'{pathIn_original}/*.png')
intermed_files = glob.glob(f'{pathIn_intermed}/*.png')
output_files = glob.glob(f'{pathIn_output}/*.png')

# for sorting the file names properly
original_files.sort(key=natural_keys)
intermed_files.sort(key=natural_keys)
output_files.sort(key=natural_keys)

frame_array = []

# for sorting the file names properly
for i, (original_filename, intermed_filename, output_filename) in enumerate(zip(original_files, intermed_files, output_files)):

    img_intermed = cv2.imread(intermed_filename)

    img_output = cv2.imread(output_filename)

    img_original = Image.open(original_filename)
    img_original.thumbnail((max(img_intermed.shape[:2]), max(img_intermed.shape[:2])), Image.ANTIALIAS)
    img_original = cv2.cvtColor(np.asarray(img_original), cv2.COLOR_RGB2BGR)

    img = np.concatenate([img_original, img_intermed, img_output], 1)
    height, width, layers = img.shape
    size = (width, height)

    # inserting the frames into an image array
    frame_array.append(img)
    print(f"{i} of {len(original_files)}")

out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()