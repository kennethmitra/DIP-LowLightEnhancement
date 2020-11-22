import torch.nn as nn
import torch.nn.functional as F
import torch
from paq2piq_standalone import InferenceModel, RoIPoolModel

import glob
from PIL import Image
import numpy as np

class ColorConstancyLoss(nn.Module):
    """
    Color Constancy Loss is defined as the sum of the squares of the difference between the mean value of every pair of channels in the output image.
    This is because in an ideal image, the average value of all pixels over all channels should approach gray (where all channels have same average value)
    i.e) SUM( (<avg of channel p> - <avg of channel q>)^2 [foreach (p,q) in {(R, G), (R, B), (G, B)}] )
    """
    def __init__(self, method, device, patch_size=16, epsilon=1e-7, gammas=(0.5, 2)):
        super(ColorConstancyLoss, self).__init__()
        self.method = method
        self.epsilon = epsilon
        self.gammas = gammas
        self.avgpool = nn.AvgPool2d(patch_size).to(device)
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='nearest').to(device)

    def forward(self, enhanced, orig=None):

        # Here's the version according to their code
        if self.method == 1:
            # Find the mean of each channel in X
            mean_r, mean_g, mean_b = torch.mean(enhanced, dim=(2, 3)).split(split_size=1, dim=1)

            # Find squared difference between each combination
            diff_rg = (mean_r - mean_g) ** 2
            diff_rb = (mean_r - mean_b) ** 2
            diff_gb = (mean_g - mean_b) ** 2
            col_loss_code = (diff_rg**2 + diff_rb**2 + diff_gb**2)**0.5
            return col_loss_code.mean()

        # Here's the version according to the math in their paper
        if self.method == 2:
            # Find the mean of each channel in X
            mean_r, mean_g, mean_b = torch.mean(enhanced, dim=(2, 3)).split(split_size=1, dim=1)

            # Find squared difference between each combination
            diff_rg = (mean_r - mean_g) ** 2
            diff_rb = (mean_r - mean_b) ** 2
            diff_gb = (mean_g - mean_b) ** 2
            col_loss_paper = diff_rg + diff_rb + diff_gb
            return col_loss_paper.mean()

        if self.method == 3:
            """
            The problem with their color constancy (white balance) loss is it muddies patches of large constant color 
            to gray like sky or buildings. This is because those images do not fit the gray world hypothesis. 
            We attempt to correct this by finding the average rgb values through an average weighted by the region's 
            variance. Thus the regions with more constant color will be weighted less.
            """

            # Gaussian Blur Image with radius 2 (To avoid blocky artifacts)

            # Get average of each region
            region_avgs = self.avgpool(enhanced)
            # Upsample averages so every pixel in region is the average
            region_avgs_upsampled = self.upsample(region_avgs)
            # Subtract average of region from each pixel and square to get variance
            variances = (enhanced - region_avgs_upsampled) ** 2
            # Average the variances within each region to get per region variance
            variances = self.avgpool(variances) + self.epsilon
            # Multiply average R, G, B of each region by their variance.
            # Then add all the regions together and divide by total variance.
            weighted_avgs = torch.sum(region_avgs * variances, dim=(2, 3)) / torch.sum(variances, dim=(2, 3))

            mean_r, mean_g, mean_b = weighted_avgs.split(split_size=1, dim=1)

            mean_r = mean_r
            mean_g = mean_g * 1.05
            mean_b = mean_b / 0.99

            diff_rg = (mean_r - mean_g) ** 2
            diff_rb = (mean_r - mean_b) ** 2
            diff_gb = (mean_g - mean_b) ** 2

            col_loss_custom = diff_rg + diff_rb + diff_gb
            return col_loss_custom.mean()

        if self.method == 4:
            assert orig is not None
            # Gamma enhance original image (To help enhance dark images)
            if self.gammas is not None:
                orig = (orig**self.gammas[0] + orig**self.gammas[1]) / 2
            # Get average of each region
            region_avgs = self.avgpool(orig)
            # Upsample averages so every pixel in region is the average
            region_avgs_upsampled = self.upsample(region_avgs)
            # Subtract average of region from each pixel and square to get variance
            variances = (orig - region_avgs_upsampled) ** 2
            # Average the variances within each region to get per region variance
            variances = self.avgpool(variances) + self.epsilon
            # Multiply average R, G, B of each region by their variance.
            # Then add all the regions together and divide by total variance.
            region_avgs = self.avgpool(enhanced)
            weighted_avgs = torch.sum(region_avgs * variances, dim=(2, 3)) / torch.sum(variances, dim=(2, 3))

            mean_r, mean_g, mean_b = weighted_avgs.split(split_size=1, dim=1)

            mean_r = mean_r
            mean_g = mean_g
            mean_b = mean_b

            diff_rg = (mean_r - mean_g) ** 2
            diff_rb = (mean_r - mean_b) ** 2
            diff_gb = (mean_g - mean_b) ** 2

            col_loss_custom = diff_rg + diff_rb + diff_gb
            return col_loss_custom.mean()

class ColorVarianceLoss(nn.Module):
    """
    This custom loss function returns the negative of the average variance in R, G, B values.
    It attempts to spread out the color histogram to make images more vivid.
    """
    def __init__(self):
        super(ColorVarianceLoss, self).__init__()

    def forward(self, X):
        col_means = torch.mean(X, dim=(2, 3), keepdim=True)
        col_variance = (X - col_means) ** 2
        col_variance = torch.mean(col_variance, dim=(1, 2, 3))
        return -col_variance.mean(dim=0)

class ExposureControlLoss(nn.Module):
    """
    The exposure control loss accounts for over/under expose regions of an image.
    Let E be the gray level in the RGB color space (Authors set E to 0.6 although they did not find significant difference for E in [0.4, 0.7])
    We also define the set of all non-overlapping square patches of size patch_size (A grid with cell size <patch_size>)
    The exposure control loss is then the average of the absolute values of the differences between the average intensity within a patch and E
    i.e) (1/<num patches>) * SUM(Y - E foreach patch) where Y is the average value of the patch
    """
    def __init__(self, method, device, gray_value=0.6, patch_size=16):
        super(ExposureControlLoss, self).__init__()
        self.gray_value = gray_value
        self.patch_size = patch_size
        self.method = method

        # AvgPool2d is used to compute the average value of all non-overlapping patches
        self.pool = nn.AvgPool2d(kernel_size=patch_size).to(device)

    def forward(self, X):
        X_grayscale = torch.mean(X, dim=1)  # Average over the color channel | Output dims: n,h,w
        patch_intensities = self.pool(X_grayscale)

        # TODO in their paper they use L1 distance but in their code they use L2 distance

        # Original Code version
        if self.method == 1:
            exp_loss_orig_code = torch.mean((patch_intensities - self.gray_value)**2)
            return exp_loss_orig_code

        # Paper version
        if self.method == 2:
            exp_loss_paper = torch.mean(torch.abs(patch_intensities - self.gray_value), dim=(0, 1, 2))
            return exp_loss_paper


class SpatialConsistencyLoss(nn.Module):
    """
    The Spatial Consistency Loss attempts to preserve the difference between neighboring regions (top, down, left right)
    i.e) (1/K) * SUM( SUM( (|Y_i - Y_j| - |I_i - I_j|)^2 ) foreach neighbor region j) over all regions i)
    """
    def __init__(self, device, method, gammas=(0.5, 2), pool_size=4):
        super(SpatialConsistencyLoss, self).__init__()
        self.method = method
        self.gammas = gammas
        self.pool = nn.AvgPool2d(pool_size).to(device)
        self.kernel_left = torch.FloatTensor([[0, 0, 0],
                                              [-1, 1, 0],
                                              [0, 0, 0]]).unsqueeze(0).unsqueeze(0).to(device)
        self.kernel_right = torch.FloatTensor([[0, 0, 0],
                                               [0, 1, -1],
                                               [0, 0, 0]]).unsqueeze(0).unsqueeze(0).to(device)
        self.kernel_up = torch.FloatTensor([[0, -1, 0],
                                            [0, 1, 0],
                                            [0, 0, 0]]).unsqueeze(0).unsqueeze(0).to(device)
        self.kernel_down = torch.FloatTensor([[0, 0, 0],
                                              [0, 1, 0],
                                              [0, -1, 0]]).unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, enhanced, original):

        # Convert to grayscale (keepdim=True since F.conv2d() expects n,c,h,w)
        if self.gammas is not None:
            original = (original ** self.gammas[0] + original ** self.gammas[1]) / 2

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

        spa_loss = torch.mean(diff_left + diff_right + diff_up + diff_down, dim=(0, 1, 2, 3))

        return spa_loss


class IlluminationSmoothnessLoss(nn.Module):
    """
    Since we predict a curve for each channel of each pixel in the input image (3 x H x W), we want to make sure that regions have smooth changes in illumination.
    We do this by computing the sum of the differences in X and Y directions of each curve in the curve map
    i.e) illumination_loss = SUM( SUM( GRAD_X(curve_map(c, i)) + GRAD_Y(curve_map(c, i)) foreach channel c) foreach iteration i in n )
    """
    def __init__(self, method):
        super(IlluminationSmoothnessLoss, self).__init__()
        self.method = method

    def forward(self, X):

        # TODO the paper's code implementation squares each element in the gradient and adds it all together, which is different from the paper

        # Their code implementation
        if self.method == 1:
            batch_size = X.size()[0]
            count_h = (X.size()[2] - 1) * X.size()[3]
            count_w = X.size()[2] * (X.size()[3] - 1)
            h_tv = torch.pow((X[:, :, 1:, :] - X[:, :, :-1, :]), 2).sum()
            w_tv = torch.pow((X[:, :, :, 1:] - X[:, :, :, :-1]), 2).sum()
            ill_loss_code = (h_tv / count_h + w_tv / count_w) / batch_size
            return ill_loss_code

        # What I think their paper meant
        if self.method == 2:
            magn_diff_x = torch.sum((X[:, :, :, 1:] - X[:, :, :, :-1])**2, dim=(2, 3))**0.5
            magn_diff_y = torch.sum((X[:, :, 1:, :] - X[:, :, :-1, :])**2, dim=(2, 3))**0.5

            ill_loss_paper = torch.sum((magn_diff_x + magn_diff_y)**2, dim=1) / (X.shape[1] * X.shape[2] * X.shape[3])
            return ill_loss_paper.mean()

        # See https://github.com/bsun0802/Zero-DCE/blob/cedd6bc1bef935727e3b15d4b328840aa1a0fca4/code/utils.py#L16
        # https://remi.flamary.com/demos/proxtv.html
        if self.method == 3:
            diff_x = ((X[:, :, :, 1:] - X[:, :, :, :-1])).abs().mean(dim=(2, 3))
            diff_y = ((X[:, :, 1:, :] - X[:, :, :-1, :])).abs().mean(dim=(2, 3))
            total_variance = (diff_x + diff_y).mean(dim=1).mean(dim=0) * 3
            return total_variance

class ImageQualityLoss(nn.Module):
    """
    Uses the PaQ-2-PiQ model to compute reference-less image quality
    See https://github.com/baidut/paq2piq/blob/master/demo.ipynb
    """
    def __init__(self, method, device, blk_size=(3, 5)):
        super(ImageQualityLoss, self).__init__()
        self.model = InferenceModel(RoIPoolModel(), './models/RoIPoolModel.pth', device=device)
        self.model.blk_size = blk_size
        self.method = method

    def forward(self, enhanced, original=None):

        if self.method == 1:
            enhanced_scores = []
            for i in range(enhanced.shape[0]):
                enhanced_scores.append(self.model.predict_global_with_grad(enhanced[i]))
            enhanced_scores = torch.stack(enhanced_scores)
            return -(torch.mean(enhanced_scores)/100)

        if self.method == 2:
            assert original is not None
            diff_scores = []
            for i in range(enhanced.shape[0]):
                diff_scores.append(self.model.predict_global_with_grad(enhanced[i]) - self.model.predict_global_with_grad(original[i]))
            diff_scores = torch.stack(diff_scores)
            return -(torch.mean(diff_scores) / 100)

if __name__ == '__main__':

    def alpha_total_variation_copied(A):
        '''
        Links: https://remi.flamary.com/demos/proxtv.html
               https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html#total_variation
        '''
        delta_h = A[:, :, 1:, :] - A[:, :, :-1, :]
        delta_w = A[:, :, :, 1:] - A[:, :, :, :-1]

        # TV used here: L-1 norm, sum R,G,B independently
        # Other variation of TV loss can be found by google search
        tv = delta_h.abs().mean((2, 3)) + delta_w.abs().mean((2, 3))
        loss = torch.mean(tv.sum(1) / (A.shape[1] / 3))
        return loss


    def alpha_total_variation_copied2(A):
        '''
        Links: https://remi.flamary.com/demos/proxtv.html
               https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html#total_variation
        '''
        diff_y = (A[:, :, 1:, :] - A[:, :, :-1, :]).abs().mean((2, 3))
        diff_x = (A[:, :, :, 1:] - A[:, :, :, :-1]).abs().mean((2, 3))

        # TV used here: L-1 norm, sum R,G,B independently
        # Other variation of TV loss can be found by google search
        tv = diff_y + diff_x
        loss = torch.mean(tv.sum(1) / (A.shape[1]))
        #loss = torch.mean(tv.sum(1) / (A.shape[1] / 3))
        return loss


    def exposure_control_loss_copied(enhances, rsize=16, E=0.6):
        avg_intensity = F.avg_pool2d(enhances, rsize).mean(1)  # to gray: (R+G+B)/3
        exp_loss = (avg_intensity - E).abs().mean()
        return exp_loss


    # Color constancy loss via gray-world assumption.   In use.
    def color_constency_loss_copied(enhances):
        plane_avg = enhances.mean((2, 3))
        col_loss = torch.mean((plane_avg[:, 0] - plane_avg[:, 1]) ** 2
                              + (plane_avg[:, 1] - plane_avg[:, 2]) ** 2
                              + (plane_avg[:, 2] - plane_avg[:, 0]) ** 2)
        return col_loss


    def spatial_consistency_loss_copied(enhances, originals, to_gray, neigh_diff, rsize=4):
        # convert to gray
        enh_gray = F.conv2d(enhances, to_gray)
        ori_gray = F.conv2d(originals, to_gray)

        # average intensity of local regision
        enh_avg = F.avg_pool2d(enh_gray, rsize)
        ori_avg = F.avg_pool2d(ori_gray, rsize)

        # calculate spatial consistency loss via convolution
        enh_pad = F.pad(enh_avg, (1, 1, 1, 1), mode='replicate')
        ori_pad = F.pad(ori_avg, (1, 1, 1, 1), mode='replicate')
        enh_diff = F.conv2d(enh_pad, neigh_diff)
        ori_diff = F.conv2d(ori_pad, neigh_diff)

        spa_loss = torch.pow((enh_diff - ori_diff), 2).sum(1).mean()
        return spa_loss


    def get_kernels(device):
        # weighted RGB to gray
        K1 = torch.tensor([0.3, 0.59, 0.1], dtype=torch.float32).view(1, 3, 1, 1).to(device)
        # K1 = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float32).view(1, 3, 1, 1).to(device)

        # kernel for neighbor diff
        K2 = torch.tensor([[[0, -1, 0], [0, 1, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
                           [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 1, -1], [0, 0, 0]]], dtype=torch.float32)
        K2 = K2.unsqueeze(1).to(device)
        return K1, K2

    # Get compute device
    seed = 42
    print("-------------------------------GPU INFO--------------------------------------------")
    print('Available devices ', torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current cuda device ', device)
    if device != "cpu":
        print('Current CUDA device name ', torch.cuda.get_device_name(device))
        torch.cuda.manual_seed(seed)
    print("-----------------------------------------------------------------------------------")


    to_gray, neigh_diff = get_kernels(device)

    # Load images
    images_list = glob.glob("./images/progress_pics/*.jpg")
    crop_size = 256
    images = []
    for image_path in images_list:
        image = Image.open(image_path)                                              # Load image
        image = image.resize((crop_size, crop_size), Image.ANTIALIAS)               # Crop to square
        image = torch.from_numpy(np.asarray(image).astype(float)).float() / 255.0   # Convert to tensor
        image = image.permute(2, 0, 1)                                              # Convert to (n,c,h,w) order
        images.append(image)



    # Convert images list to tensor
    images = torch.stack(images).to(device)
    print(f"Input dimensions: {list(images.shape)}")

    colLoss = ColorConstancyLoss(method=2, device=device)
    result = colLoss(images)
    result2 = color_constency_loss_copied(images)
    print(f"Color Constancy Loss: {result}")
    print(f"Copied Color Constancy Loss: {result2}")

    expLoss = ExposureControlLoss(method=2, device=device)
    result = expLoss(images)
    result2 = exposure_control_loss_copied(images)
    print(f"Exposure Control Loss: {result}")
    print(f"Copied Exposure Control Loss: {result2}")

    spaLoss = SpatialConsistencyLoss(device=device)
    result = spaLoss(images, torch.stack([images[0], images[0], images[0], images[0], images[0]]))
    result2 = spatial_consistency_loss_copied(images, torch.stack([images[0], images[0], images[0], images[0], images[0]]), neigh_diff=neigh_diff, to_gray=to_gray)
    print(f"Spatial Consistency Loss: {result}")
    print(f"Copied Spatial Consistency Loss: {result2}")

    illLoss = IlluminationSmoothnessLoss(method=3)
    result = illLoss(images)
    result2 = alpha_total_variation_copied(images)
    result3 = alpha_total_variation_copied2(images)
    print(f"Illumination Smoothness Loss: {result}")
    print(f"Their Illumination Smoothness Loss: {result2}")
    print(f"Their Illumination Smoothness Loss2: {result3}")


    print()
    print()
    colVarLoss = ColorVarianceLoss()
    result = colVarLoss(images)
    print(f"Col Var Loss: {result}")

    print()
    print()
    customWB = ColorConstancyLoss(method=3, patch_size=16, device=device)
    result = customWB(images)
    print(f"Custom WB: {result}")

    print()
    print()
    imageQualLoss = ImageQualityLoss(method=1)
    result = imageQualLoss(images)
    print(f"IQA Loss: {result}")

    print()
    print()
    imageQualLoss = ImageQualityLoss(method=2)
    result = imageQualLoss(enhanced=images, original=torch.stack([images[0], images[0], images[3], images[3], images[3]]))
    print(f"IQA Loss: {result}")

