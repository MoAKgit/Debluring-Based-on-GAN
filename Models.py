# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 19:22:11 2022

@author: Mohammad
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import networks
import os
from util import PSNR, SSIM
import util
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from losses import *


class Model():
    def __init__(self, args, device):
        
        
        self.args = args
        use_sigmoid = args.gan_type == 'gan'
        self.netG1 = networks.define_G(args, device )
        self.netD1 = networks.define_D(args, device)
        
            
        self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(), 
                                             lr=args.learning_rate)
        self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(), 
                                             lr=args.learning_rate)

        
        self.real_label = 1.0
        self.fake_label = 0.0
        self.device = device
        self.Tensor = tensor=torch.FloatTensor
        
        self.loss_mse = nn.MSELoss()
        self.loss_perceptual = PerceptualLoss().cuda()
        self.Laplacian_edge_loss = Laplacian_edge_loss(3)
        self.sobel_edge_loss = sobel_edge_loss(3)
        self.SSIM_loss0 = SSIMLoss(3)
        self.SSIM_loss1 = SSIMLoss(7)
        self.SSIM_loss2 = SSIMLoss(11)
        self.SSIM_loss3 = SSIMLoss(15)
        self.SSIM_loss4 = SSIMLoss(21)
        
        self.HaarPSILoss = HaarPSILoss()

    def optimization(self, real_A, real_B):
        fake_A1 = self.netG1(real_A )
        pred_fake_A1 = self.netD1(fake_A1)
        pred_real_B1 = self.netD1(real_B)
        
        ####  updating D1
        for i in range(1):
            
            b_size = real_A.shape[0]
            real_img_label = self.Tensor(
                pred_real_B1.size()).fill_(self.real_label).to(self.device)
            errD_real = self.loss_mse(pred_real_B1, real_img_label)
            
            fake_img_label = self.Tensor(
                pred_fake_A1.size()).fill_(self.fake_label).to(self.device)
            errD_fake = self.loss_mse(pred_fake_A1, fake_img_label)
        
        loss = errD_real + errD_fake
        self.optimizer_D1.zero_grad()
        loss.backward()
        self.optimizer_D1.step()
        
        ##### updating G1
        fake_A1 = self.netG1(real_A )
        pred_fake_A1 = self.netD1(fake_A1)
        
        fake_img_label = self.Tensor(
            pred_fake_A1.size()).fill_(self.real_label).to(self.device)
        
        errG_fake = self.loss_mse(pred_fake_A1, fake_img_label)
        perceptual_loss = self.loss_perceptual(fake_A1, real_B)
        lossContent = self.loss_mse(fake_A1, real_B) 
        Laplacian_edge_loss = self.Laplacian_edge_loss(fake_A1, real_B)
        sobel_edge_loss = self.sobel_edge_loss(fake_A1, real_B)
        SSIM_loss0 = self.SSIM_loss0(fake_A1, real_B)
        SSIM_loss1 = self.SSIM_loss1(fake_A1, real_B)
        SSIM_loss2 = self.SSIM_loss2(fake_A1, real_B)
        SSIM_loss3 = self.SSIM_loss3(fake_A1, real_B)
        SSIM_loss4 = self.SSIM_loss4(fake_A1, real_B)
        
        HaarpsiLoss = self.HaarPSILoss(fake_A1, real_B)
        
        SSIM_loss = SSIM_loss0 + SSIM_loss1 +SSIM_loss2 + SSIM_loss3+ SSIM_loss4
        
        loss = errG_fake + lossContent+ perceptual_loss + Laplacian_edge_loss +\
            sobel_edge_loss  +  SSIM_loss + HaarpsiLoss       
        self.optimizer_G1.zero_grad()
        loss.backward()
        self.optimizer_G1.step()
        


    def test(self, model, test_dataloader, device, epoch):
        
        psnrMetric = []
        nun_samples = 0
        for i, (real_A, real_B) in enumerate(test_dataloader,0):
            real_A, real_B= map(lambda x: x.to(device), [real_A, real_B])
            b,_,_,_ = real_B.shape
            fake_A1 = model(real_A)
            img = torch.cat( (real_A[0], fake_A1[0], real_B[0]), dim= 2 )
            img = util.tensor2im(img)
            
            fake_A1 = util.tensor2im(fake_A1[0])
            real_B = util.tensor2im(real_B[0])
            psnrMetric.append( PSNR(fake_A1, real_B) )
            nun_samples += 1
            if epoch == self.args.epochs-1:
                if not os.path.isdir('results/imgs'):
                    os.makedirs('results/imgs')
                
                img_path = 'results/imgs/img_%d_%d.png' % (epoch,i)
                
                util.save_img(img, img_path)
        
        psnrMetric = np.asarray(psnrMetric)
        psnr = psnrMetric.mean()
        
        
        return psnr


    def save_model(self, epoch = 'latest'):
        self.save_network(self.netG1, epoch, 'G')
        self.save_network(self.netD1, epoch, 'D')
        
        
    def save_network(self, network, epoch, network_label):
        save_filename = '%s_net_%s.pth' % (epoch, network_label)
        if not os.path.isdir(self.args.checkpoints_dir):
            os.makedirs(self.args.checkpoints_dir)
            
        save_path = os.path.join(self.args.checkpoints_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        
        if torch.cuda.is_available():
            network.cuda(device=self.device)


    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))