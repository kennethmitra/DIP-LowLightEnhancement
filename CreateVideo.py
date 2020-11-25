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

pathIn_enhanced = './images/video1_output/selected'
pathIn_original = './images/videos/video1'
pathOut = 'enhanced2.avi'
fps = 25


enhanced_files = glob.glob(f'{pathIn_enhanced}/*.png')
original_files = glob.glob(f'{pathIn_original}/*.png')
# for sorting the file names properly
enhanced_files.sort(key=natural_keys)
original_files.sort(key=natural_keys)

frame_array = []
# enhanced_files = [f for f in os.listdir(pathIn_enhanced) if isfile(join(pathIn_enhanced, f))]
# enhanced_files.sort(key=natural_keys)

# for sorting the file names properly
for i, (original_filename, enhanced_filename) in enumerate(zip(original_files, enhanced_files)):
    # reading each enhanced_files
    img_enhanced = cv2.imread(enhanced_filename)
    img_original = Image.open(original_filename)
    img_original.thumbnail((max(img_enhanced.shape[:2]), max(img_enhanced.shape[:2])), Image.ANTIALIAS)
    img_original = cv2.cvtColor(np.asarray(img_original), cv2.COLOR_RGB2BGR)
    img = np.concatenate([img_original, img_enhanced], 1)
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