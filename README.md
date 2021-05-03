# DIP-LowLightEnhancement

Link to our presentation describing the project:
https://www.youtube.com/watch?v=owQkejcJhro

## File Descriptions
Model.py - Model outputting pixelwise RGB curves and enhanced image

LossFunctions.py - Defines functions based on various metrics to compute loss of an enhanced image without a reference image

Train.py - Script used to train the model

Test.py - Script used to evaluate model on test data

CreateFrames.py - Splits a video into frames

EnhanceFrames.py - Applies low light enhancement to frames of video (using several versions of the model + an image quality net to select the best image)

CreateVideo.py - String together enhanced frames into a video
