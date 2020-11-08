import torch.nn as nn
import torch

import glob
from PIL import Image
import numpy as np

class ColorConstancyLoss(nn.Module):
    """
    Color Constancy Loss is defined as the sum of the squares of the difference between the mean value of every pair of channels in the output image.
    This is because in an ideal image, the average value of all pixels over all channels should approach gray (where all channels have same average value)
    i.e) SUM( (<avg of channel p> - <avg of channel q>)^2 [foreach (p,q) in {(R, G), (R, B), (G, B)}] )
    """
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, X):

        # Find the mean of each channel in X
        mean_r, mean_g, mean_b = torch.mean(X, dim=(2, 3))

        # Find squared difference between each combination
        diff_rg = (mean_r - mean_g)**2
        diff_rb = (mean_r - mean_b)**2
        diff_gb = (mean_g - mean_b)**2

        # TODO examine which of the following versions perform better

        # Here's the version according to their code
        col_loss_code = (diff_rg**2 + diff_rb**2 + diff_gb**2)**0.5

        # Here's the version according to the math in their paper
        col_loss_paper = diff_rg + diff_rb + diff_gb

        return col_loss_code, col_loss_paper


class ExposureControlLoss(nn.Module):
    """
    The exposure control loss accounts for over/under expose regions of an image.
    Let E be the gray level in the RGB color space (Authors set E to 0.6 although they did not find significant difference for E in [0.4, 0.7])
    We also define the set of all non-overlapping square patches of size patch_size (A grid with cell size <patch_size>)
    The exposure control loss is then the average of the absolute values of the differences between the average intensity within a patch and E
    i.e) (1/<num patches>) * SUM(Y - E foreach patch) where Y is the average value of the patch
    """
    def __init__(self, gray_value=0.6, patch_size=16):
        super(ExposureControlLoss, self).__init__()
        self.gray_value = gray_value
        self.patch_size = patch_size

        # AvgPool2d is used to compute the average value of all non-overlapping patches
        self.pool = nn.AvgPool2d(kernel_size=patch_size)

    def forward(self, X):
        X_grayscale = torch.mean(X, dim=1)  # Average over the color channel | Output dims: n,h,w
        patch_intensities = self.pool(X_grayscale)

        # TODO in their paper they use L1 distance but in their code they use L2 distance
        # TODO in the code they average over all images in the batch as well. How tf their code even work?

        # Original Code version
        exp_loss_orig_code = torch.mean((patch_intensities - self.gray_value)**2)

        # What I think the code meant to do version
        exp_loss_fixed_code = torch.mean((patch_intensities - self.gray_value)**2, dim=(1, 2))

        # Paper version
        exp_loss_paper = torch.mean(torch.abs(patch_intensities - self.gray_value), dim=(1, 2))

        return exp_loss_orig_code, exp_loss_fixed_code, exp_loss_paper



if __name__ == '__main__':

    # Load images
    images_list = glob.glob("./images/*.jpg")
    crop_size = 256
    images = []
    for image_path in images_list:
        image = Image.open(image_path)                                              # Load image
        image = image.resize((crop_size, crop_size), Image.ANTIALIAS)               # Crop to square
        image = torch.from_numpy(np.asarray(image).astype(float)).float() / 255.0   # Convert to tensor
        image = image.permute(2, 0, 1)                                                      # Convert to (n,c,h,w) order
        images.append(image)

    # Convert images list to tensor
    images = torch.stack(images)
    print(f"Input dimensions: {list(images.shape)}")

    colLoss = ColorConstancyLoss()
    result = colLoss(images)
    print(f"Color Constancy Loss: {result}")

    expLoss = ExposureControlLoss()
    result = expLoss(images)
    print(f"Exposure Control Loss: {result}")
