# SSIM: Structural Similarity Index
# 2004: https://ece.uwaterloo.ca/~z70wang/research/ssim/
# ref: Pypi pytorch-ssim package
#   - Compares two images and determines the degree to which 
#      they are similar. 1 if identical, 0 if completely distinct in every direction
#
# August 2019

from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import cv2

def ssim(image1, image2, K, window_size, L):
    _, channel1, _, _ = image1.size()
    _, channel2, _, _ = image2.size()
    channel = min(channel1, channel2)

    # gaussian window generation
    sigma = 1.5      # default
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    _1D_window = (gauss/gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
            
    # define constants
    # * L = 255 for constants doesn't produce meaningful results; thus L = 1
    # C1 = (K[0]*L)**2;
    # C2 = (K[1]*L)**2;
    C1 = K[0]**2;
    C2 = K[1]**2;
    
    mu1 = F.conv2d(image1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(image2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(image1*image1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(image2*image2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(image1*image2, window, padding = window_size//2, groups = channel) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

if __name__=="__main__":
    # opencv image load
    I1 = cv2.imread('./image.png')
    I2 = cv2.imread('./mixed.png')
    # I2 = cv2.imread('./blur.png')
    I2 = cv2.resize(I2, I1.shape[0:2])
    # print(I1.shape, I2.shape) # returns (256,256,3)
    
    # tensors
    I1 = torch.from_numpy(np.rollaxis(I1, 2)).float().unsqueeze(0)/255.0
    I2 = torch.from_numpy(np.rollaxis(I2, 2)).float().unsqueeze(0)/255.0
    # print(I1.size(), I2.size()) # returns torch([1,3,256,256])
    
    # tensor.autograd.Variable (Automatic differentiation variable)
    I1 = Variable(I1, requires_grad = True)
    I2 = Variable(I2, requires_grad = True)
    
    # default constants
    K = [0.01, 0.03]
    L = 255; 
    window_size = 11
    
    ssim_value = ssim(I1, I2, K, window_size, L)
    
    print(ssim_value.data)
