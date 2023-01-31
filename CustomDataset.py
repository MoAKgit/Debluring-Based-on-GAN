# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:55:01 2022

@author: Mohammad
"""
from torch.utils.data import DataLoader,Dataset
import glob as glob
import os
import numpy as np
from torchvision import transforms
import random
from PIL import Image
import torch
import cv2
from PIL import Image
import math
import torch.nn.functional as F
import torch.nn.functional as nnf

class Gopro(Dataset):
    def __init__(self, args, phase = None ,transform=True):
        super(Gopro, self).__init__()
        self.args = args
        self.phase = phase
        if phase == 'train':
            self.path =  args.train_path
        elif phase == 'test':
            self.path = args.test_path
        else:
            raise 'Path is not recognized'
            
        self.image_paths = glob.glob(os.path.join(self.path, "*.png"))
        
        if self.phase == 'test':
            self.image_paths = self.image_paths[:1000]
        
        print('Number of imgs: %d'%(len(self.image_paths)))
        
        self.transform = transforms.Compose([transforms.ToTensor(), 
                                             transforms.Normalize( (0.5, 0.5, 0.5),
                                                                  (0.5, 0.5, 0.5) )])
        
        
    def __getitem__(self, index):
        
        img_path = self.image_paths[index]
        
        img = Image.open(img_path).convert('RGB')
        # img = cv2.imread(img_path)

        img = self.transform(img).float()
        w_total = img.size(2)
        w = int(w_total / 2)
        h = img.size(1)
        
        
        
        if self.phase == 'train':
            
            A = img[:, :, :w]
            
            B = img[:, :, w:]
            
            if (self.args.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A = A.index_select(2, idx)
                B = B.index_select(2, idx)
                
        if self.phase == 'test':
            
            A = img[:, :, :w]
            B = img[:, :, w:]
        
        
        return A, B
    
    def __len__(self):
        return len(self.image_paths)
    
    
    
    
    
    
    
# class SiameseNetworkDataset(Dataset):
    
#     def __init__(self, path ,transform=True):
#         self.path = path    
#         self.feedshape = [1, 28, 28]
#         self.image_paths = glob.glob(os.path.join(self.path, "*.png"))
        
        
#         print('Number of imgs: %d'%(len(self.image_paths)))
#         self.labels = np.load(self.path+"/bbox.npy")
#         self.indice = np.arange(len(self.image_paths))
#         print(self.labels.shape)
#         self.transform = transforms.Compose([transforms.ToTensor(),
#                                               transforms.Resize(self.feedshape[1:])])
#     def __getitem__(self,index):
        
#         idx0 = random.choice(self.indice)
#         label0 = self.labels[idx0,4]
        
#         should_get_same_class = random.randint(0,1) 
#         if should_get_same_class:
#             while True:
#                 idx1 = random.choice(self.indice)
#                 if label0 == self.labels[idx1,4]:
#                     break
#         else:
#             idx1 = random.choice(self.indice)
        
#         label1 = self.labels[idx1,4]
#         # img0 = Image.open(self.image_paths[idx0]).convert("RGB")
#         # img1 = Image.open(self.image_paths[idx1]).convert("RGB")
        
#         img0 = Image.open(self.image_paths[idx0])
#         img1 = Image.open(self.image_paths[idx1])

#         # img0 = img0.convert("L")
#         # img1 = img1.convert("L")        

#         img0 = self.transform(img0).float()
#         img1 = self.transform(img1).float()

#         # return img0, img1, torch.FloatTensor([label0 == label1 ])
#         return img0, img1, torch.FloatTensor([int(label0 == label1 )])
    
#     def __len__(self):
#         return len(self.image_paths)
    