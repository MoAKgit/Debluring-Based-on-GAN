# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 15:31:43 2021

@author: Mohammad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.signal import savgol_filter

plt.figure(1)

# plt.subplot(3,1,1)
# data = pd.read_csv("PSNR_history.csv")
# plt.plot(data[0:])
# plt.grid()


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_ = np.squeeze(y, 1)
    print(y_.shape)
    y_smooth = np.convolve(y_, box, mode='same')
    return y_smooth

# plt.subplot(3,1,2)
# data = pd.read_csv("PSNR_history1.csv") 
# plt.plot(data[0:])
# # plt.ylim([0.008,0.04])
# plt.grid()

# def smooth(y, box_pts):
#     box = np.ones(box_pts)/box_pts03/2103 [19:09<0
#     y_smooth = np.convolve(y, box, mode='same')
#     return y_smooth

data = pd.read_csv("PSNR_history0.csv")
data1 = pd.read_csv("PSNR_history.csv")

# data = smooth(data,4)
# data1 = smooth(data1,4)

# data1 = np.squeeze(data1, 1)


# plt.plot(data, 'b', label='DebluredGAN')

plt.plot(data1, 'r', label='New Model')

plt.legend()
plt.grid()
# plt.ylim([28,32])
# plt.xlim([0,200])
# [b,c] = plt.plot([data,data1])
# plt.legend([b,c], ["b","c"], loc=0)
# plt.show()