import torch.nn as nn
import torch.nn.functional as F
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

class SpatialConsistencyLoss(nn.Module):
    """
    The Spatial Consistency Loss attempts to preserve the difference between neighboring regions (top, down, left right)
    i.e) (1/K) * SUM( SUM( (|Y_i - Y_j| - |I_i - I_j|)^2 ) foreach neighbor region j) over all regions i)
    """
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
        self.pool = nn.AvgPool2d(4)
        self.kernel_left = torch.FloatTensor([[0, 0, 0],
                                              [-1, 1, 0],
                                              [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        self.kernel_right = torch.FloatTensor([[0, 0, 0],
                                               [0, 1, -1],
                                               [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        self.kernel_up = torch.FloatTensor([[0, -1, 0],
                                            [0, 1, 0],
                                            [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        self.kernel_down = torch.FloatTensor([[0, 0, 0],
                                              [0, 1, 0],
                                              [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

    def forward(self, enhanced, original):

        # Convert to grayscale (keepdim=True since F.conv2d() expects n,c,h,w)
        orig_grayscale = torch.mean(original, dim=1, keepdim=True)
        enhanced_grayscale = torch.mean(enhanced, dim=1, keepdim=True)

        # Compute average of each region
        orig_regions = self.pool(orig_grayscale)
        enhanced_regions = self.pool(enhanced_grayscale)

        # Compute differences from neighbors for each region
        conv_diff_left_orig = F.conv2d(orig_regions, self.kernel_left, padding=1)
        conv_diff_right_orig = F.conv2d(orig_regions, self.kernel_right, padding=1)
        conv_diff_up_orig = F.conv2d(orig_regions, self.kernel_up, padding=1)
        conv_diff_down_orig = F.conv2d(orig_regions, self.kernel_down, padding=1)

        conv_diff_left_enhanced = F.conv2d(enhanced_regions, self.kernel_left, padding=1)
        conv_diff_right_enhanced = F.conv2d(enhanced_regions, self.kernel_right, padding=1)
        conv_diff_up_enhanced = F.conv2d(enhanced_regions, self.kernel_up, padding=1)
        conv_diff_down_enhanced = F.conv2d(enhanced_regions, self.kernel_down, padding=1)

        # Compute difference between differences in each region
        # TODO the paper says to take absolute value here but their code does not (and it makes more sense not to)
        diff_left = (conv_diff_left_orig - conv_diff_left_enhanced)**2
        diff_right = (conv_diff_right_orig - conv_diff_right_enhanced)**2
        diff_up = (conv_diff_up_orig - conv_diff_up_enhanced)**2
        diff_down = (conv_diff_down_orig - conv_diff_down_enhanced)**2

        spa_loss = torch.mean(diff_left + diff_right + diff_up + diff_down, dim=(1, 2, 3))

        return spa_loss


class IlluminationSmoothnessLoss(nn.Module):
    """
    Since we predict a curve for each channel of each pixel in the input image (3 x H x W), we want to make sure that regions have smooth changes in illumination.
    We do this by computing the sum of the differences in X and Y directions of each curve in the curve map
    i.e) illumination_loss = SUM( SUM( GRAD_X(curve_map(c, i)) + GRAD_Y(curve_map(c, i)) foreach channel c) foreach iteration i in n )
    """
    def __init__(self):
        super(IlluminationSmoothnessLoss, self).__init__()

    def forward(self, X):

        # TODO the paper's code implementation squares each element in the gradient and adds it all together, which is different from the paper

        # Their code implementation
        batch_size = X.size()[0]
        count_h = (X.size()[2] - 1) * X.size()[3]
        count_w = X.size()[2] * (X.size()[3] - 1)
        h_tv = torch.pow((X[:, :, 1:, :] - X[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((X[:, :, :, 1:] - X[:, :, :, :-1]), 2).sum()
        ill_loss_code = (h_tv / count_h + w_tv / count_w) / batch_size

        # What I think their paper meant
        magn_diff_x = torch.sum((X[:, :, :, 1:] - X[:, :, :, :-1])**2, dim=(2, 3))**0.5
        magn_diff_y = torch.sum((X[:, :, 1:, :] - X[:, :, :-1, :])**2, dim=(2, 3))**0.5

        ill_loss_paper = torch.sum((magn_diff_x + magn_diff_y)**2, dim=1) / (X.shape[1] * X.shape[2] * X.shape[3])

        return ill_loss_code, ill_loss_paper


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

    spaLoss = SpatialConsistencyLoss()
    result = spaLoss(images, torch.stack([images[0], images[0], images[0]]))
    print(f"Spatial Consistency Loss: {result}")

    illLoss = IlluminationSmoothnessLoss()
    result = illLoss(images)
    print(f"Illumination Smoothness Loss: {result}")
