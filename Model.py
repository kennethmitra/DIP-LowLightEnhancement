import torch.nn as nn
import torch.nn.functional as F
import torch


class EnhancerModel(nn.Module):

    def __init__(self):
        super(EnhancerModel, self).__init__()

        number_f = 32
        self.conv1 = nn.Conv2d(in_channels=3,            out_channels=number_f, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=number_f,     out_channels=number_f, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=number_f,     out_channels=number_f, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=number_f,     out_channels=number_f, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=number_f * 2, out_channels=number_f, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(in_channels=number_f * 2, out_channels=number_f, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7 = nn.Conv2d(in_channels=number_f * 2, out_channels=24,       kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, X):
        X1 = F.relu(self.conv1(X))
        X2 = F.relu(self.conv2(X1))
        X3 = F.relu(self.conv3(X2))
        X4 = F.relu(self.conv4(X3))

        resInput = torch.cat([X3, X4], dim=1)
        X5 = F.relu(self.conv5(resInput))
        resInput = torch.cat([X2, X5], dim=1)
        X6 = F.relu(self.conv6(resInput))
        resInput = torch.cat([X1, X6], dim=1)
        X7 = torch.tanh(self.conv7(resInput))
        return X7

    def enhance_image(self, image, curves):
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(curves, split_size_or_sections=3, dim=1)

        
        image = image + r1*(image**2-image)
        image = image + r2*(image**2-image)
        image = image + r3*(image**2-image)
        image = image + r4*(image**2-image)		
        image = image + r5*(image**2-image)		
        image = image + r6*(image**2-image)	
        image = image + r7*(image**2-image)
        image = image + r8*(image**2-image)

        # image is now = to enhanced image 
        return image
