# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:07:35 2022

@author: Mohammad
"""


import torch # pytorch
import torch.nn as nn 
import torchvision
import torchvision.transforms as trans
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import os
import numpy as np
from CustomDataset import Gopro
from Models import *
import argparse
import util
from PIL import Image
from util import PSNR, SSIM
from tqdm import tqdm
def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description='DebluredGAN')
    parser.add_argument("--epochs",dest= 'epochs', default= 1000) 
    parser.add_argument("--batch_size",dest= 'batch_size', default= 8) 
    parser.add_argument("--learning_rate",dest= 'learning_rate', default= 0.00001) 
    parser.add_argument("--train_path",dest= 'train_path', 
                        default= "F:\data/GOPRO\AB/train")
    parser.add_argument("--test_path",dest= 'test_path', 
                        default= "F:\data/GOPRO\AB/test")
    parser.add_argument("--input_nc",dest= 'input_nc', default= 3) 
    parser.add_argument("--output_nc",dest= 'output_nc', default= 3)
    parser.add_argument("--ngf",dest= 'ngf', default= 64)
    parser.add_argument("--ndf",dest= 'ndf', default= 64)
    parser.add_argument("--norm",dest= 'norm', default= 'instance')
    parser.add_argument("--n_layers_D",dest= 'n_layers_D', default= 3)
    parser.add_argument("--no_dropout",dest= 'no_dropout', default= 'store_true')
    parser.add_argument("--fineSize",dest= 'fineSize', default= 256)
    parser.add_argument("--no_flip",dest= 'no_flip', default= True)
    parser.add_argument("--ngpu",dest= 'ngpu', default= 1)
    parser.add_argument("--lambda_A",dest= 'lambda_A', default= 100)
    parser.add_argument("--gan_type",dest= 'gan_type', default= 'gan', 
                        help='wgan-gp, lsgan,gan')
    parser.add_argument("--learn_residual",dest= 'learn_residual', default= True)
    parser.add_argument("--nThreats",dest= 'nThreats', default= 3)
    parser.add_argument("--checkpoints_dir",dest= 'checkpoints_dir', default= 'ckpt')
    parser.add_argument("--continue_training",dest= 'continue_training', default= True)
    parser.add_argument("--use_dropout",dest= 'use_dropout', default= False)
    return parser.parse_args()


if __name__ == '__main__':
    print('Main')
    args = arg_parse()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)
    
    train_dataset   = Gopro(args, phase = 'train')  
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size= args.batch_size, 
                                  pin_memory= True,
                                  shuffle = True ,
                                  num_workers= args.nThreats)
    
    test_dataset   = Gopro(args, phase = 'test') 
    test_dataloader = DataLoader(test_dataset, 
                                  batch_size= 1, 
                                  pin_memory= True,
                                  shuffle = True ,
                                  num_workers= args.nThreats)
    
    model = Model(args, device)
    
    loss_history = []
    PSNR_history = []
    itr = 0
    best_psnr = 0
    for epoch in range(args.epochs):
        for i, (real_A, real_B) in enumerate(tqdm(train_dataloader),0):
            real_A, real_B= map(lambda x: x.to(device), [real_A, real_B])
            
            b,_,_,_ = real_B.shape
            model.optimization(real_A, real_B)
            itr += 1
            if itr%1000 == 0:
                fake_A1 = model.netG1(real_A)
                
                img = torch.cat( (real_A[0], fake_A1[0], real_B[0]), dim= 2 )
                img = util.tensor2im(img)
                if not os.path.isdir('results/img'):
                    os.makedirs('results/img')
                
                img_path = 'results/img/real_fake_%d_%d.png' % (epoch,i)
                util.save_img(img, img_path)
                # print('epoch: ', epoch, 'itr: ', itr)

            # psnr  = model.test(model.netG1, test_dataloader, device, epoch)
            # print('epoch: ', epoch, 'PSNR : ', psnr)
            
        if epoch %1 ==0 and epoch>=0 :
            psnr  = model.test(model.netG1, test_dataloader, device, epoch)
            PSNR_history.append(psnr)
            print('epoch: ', epoch, 'PSNR : ', psnr)
            if psnr > best_psnr:
                model.save_model()
                np.savetxt('PSNR_history.csv', PSNR_history)
    
