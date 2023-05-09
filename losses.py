# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 20:59:37 2022

@author: Mohammad
"""

import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.autograd as autograd
import numpy as np
import torchvision.models as models

from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
from torchvision.models import vgg16,vgg19 
# import kornia
from piqa import SSIM
from piqa import HaarPSI

class SSIMLoss(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.SSIM = SSIM( window_size = window_size).cuda()
        self.HaarPSI = HaarPSI().cuda()
        
    def forward(self, x, y):
        x = (x+1)/2
        y = (y+1)/2
        return 1-self.SSIM(x, y) + (1-self.HaarPSI(x, y)) 

class HaarPSILoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.HaarPSI = HaarPSI().cuda()
        
    def forward(self, x, y):
        x = (x+1)/2
        y = (y+1)/2
        return (1-self.HaarPSI(x, y)) 
    
class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()


def perceptual_loss(x, y):
    return F.mse_loss(x, y)
def PerceptualLoss():
    return FeatureLoss(perceptual_loss, [0,1,2,3,4], [1., 1., 1., 1., 0.0])

class FeatureLoss(nn.Module):
    def __init__(self, loss, blocks, weights ):
        super().__init__()
        assert all(isinstance(w, (int, float)) for w in weights)
        assert len(weights) == len(blocks)
        self.feature_loss = loss
        self.weights = torch.tensor(weights).cuda()
        # VGG16 contains 5 blocks - 3 convolutions per block and 3 dense layers towards the end
        assert len(blocks) <= 5
        assert all(i in range(5) for i in blocks)
        assert sorted(blocks) == blocks

        vgg = vgg16(pretrained=True).features
        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        vgg = vgg.cuda()

        bns = [i - 2 for i, m in enumerate(vgg) if isinstance(m, nn.MaxPool2d)]
        ## bns = [2, 7, 14, 21, 28]
        assert all(isinstance(vgg[bn], nn.Conv2d) for bn in bns)

        self.hooks = [FeatureHook(vgg[bns[i]]) for i in blocks]
        
        self.features = vgg[0: bns[blocks[-1]] + 1]
        
    def forward(self, inputs, targets):

        # # normalize foreground pixels to ImageNet statistics for pre-trained VGG
        # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # inputs = F.normalize(inputs, mean, std)
        # targets = F.normalize(targets, mean, std)

        # extract feature maps
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        loss = 0.0
        # compare their weighted loss
        for lhs, rhs, w in zip(input_features, target_features, self.weights):
            lhs = lhs.cuda()
            rhs = rhs.cuda()
#            lhs = gram_matrix(lhs)
#            rhs = gram_matrix(rhs)

            loss += self.feature_loss(lhs, rhs) * w
            
            
            
        return loss
          

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class Laplacian_edge_loss(nn.Module):
    def __init__(self, channels = 3):
        super(Laplacian_edge_loss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        kernel_laplacian = torch.matmul(k.t(),k).unsqueeze(0)
        # print('self.kernel_laplacian', kernel_laplacian.shape)
        self.kernel_laplacian = kernel_laplacian.repeat(channels,1,1,1)
        # print('self.kernel_laplacian', self.kernel_laplacian.shape)
        
        # self.kernel_laplacian = torch.Tensor([[1, 1, 2, 1, 1],
        #                                       [1, 1, 2, 1, 1],
        #                                       [2, 2, 4, 2, 2],
        #                                       [1, 1, 2, 1, 1],
        #                                       [1, 1, 2, 1, 1]])
        
        # self.kernel_laplacian = self.kernel_laplacian / torch.sum(self.kernel_laplacian)
        
        if torch.cuda.is_available():
            self.kernel_laplacian = self.kernel_laplacian.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel_laplacian.shape
        
        img1 = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        img2 = F.pad(img, ((kw+4)//2, (kh+4)//2, (kw+4)//2, (kh+4)//2), mode='replicate')
        return 0.5*F.conv2d(img1, self.kernel_laplacian, groups=n_channels) +\
            0.5*F.conv2d(img2, self.kernel_laplacian, groups=n_channels, dilation= 2)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss_laplacian = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss_laplacian 
    
    
    
class sobel_edge_loss(nn.Module):
    def __init__(self, in_channels):
        super(sobel_edge_loss, self).__init__()
        self.k_v =  torch.Tensor([[1, 0, 0, 0, -1],
                                  [1, 0, 0, 0, -1],
                                  [2, 0, 0, 0, -2],
                                  [1, 0, 0, 0, -1],
                                  [1, 0, 0, 0, -1]])
        
        self.k_h =  torch.Tensor([[-1, -1, -2, -1, -1],
                                  [ 0,  0,  0,  0,  0],
                                  [ 0,  0,  0,  0,  0],
                                  [ 0,  0,  0,  0,  0],
                                  [ 1,  1,  2,  1,  1]])
        
        self.kernel_sobel_v = self.k_v.unsqueeze(0).repeat(in_channels,1,1,1)
        self.kernel_sobel_h = self.k_h.unsqueeze(0).repeat(in_channels,1,1,1)
        
        if torch.cuda.is_available():
            self.kernel_sobel_v = self.kernel_sobel_v.cuda()
            self.kernel_sobel_h = self.kernel_sobel_h.cuda()
        
        self.loss = CharbonnierLoss()
        
    def conv_gauss(self, img):
        n_channels, _ , kw, kh = self.kernel_sobel_v.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return 0.5*F.conv2d(img, self.kernel_sobel_v, groups=n_channels) + \
            0.5*F.conv2d(img, self.kernel_sobel_h, groups=n_channels)
        
    def sobel_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff
    
    def forward(self, x, y):
        loss_sobel = self.loss(self.sobel_kernel(x), self.sobel_kernel(y))
        return loss_sobel
