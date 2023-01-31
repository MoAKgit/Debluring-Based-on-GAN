# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 16:11:10 2022

@author: Mohammad
"""


import torch
# import torch.nn as nn
from torch import nn
import functools
# from torch.autograd import Variable
import numpy as np
import os
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.002)
        # torch.nn.init.xavier_uniform_(m.weight)
        m.weight.data.uniform_(-0.002, 0.002)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        # torch.nn.init.xavier_uniform_(m.weight)
        # m.weight.data.normal_(1.0, 0.002)
        m.weight.data.uniform_(-0.002, 0.002)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(args, device ):
    
    netG = None
    norm_layer = get_norm_layer(norm_type = args.norm)
    
    netG = scaled_ResnetGenerator(args,norm_layer=norm_layer, n_blocks=4)
    
    checkpoints_path = 'ckpt/latest_net_G.pth'
    if args.continue_training == True:
        if os.path.isfile(checkpoints_path):
            netG.load_state_dict(torch.load(checkpoints_path))
            print('checkpoints netG loaded:)')
        else:
            print('No checkpoints!')
            netG.apply(weights_init)
            print('Weights are randomly initialized')
    else:
        netG.apply(weights_init)
        print('Weights are randomly initialized')
    
    
    return netG.to(device)

def define_D(args, device):
    netD = None
    
    
    norm_layer = get_norm_layer(norm_type=args.norm)


    netD = NLayerDiscriminator(args, 3, norm_layer=norm_layer)

    checkpoints_path = 'ckpt/latest_net_D.pth'
    if args.continue_training == True:
        if os.path.isfile(checkpoints_path):
            netD.load_state_dict(torch.load(checkpoints_path))
            print('checkpoints netD loaded:)')
        else:
            print('No checkpoints!')
            netD.apply(weights_init)
            print('Weights are randomly initialized')
    else:
        netD.apply(weights_init)
        print('Weights are randomly initialized')
    
    return netD.to(device)

class New_att(nn.Module):
    def __init__(self):
        super(New_att, self).__init__()
        self.window_size = (8,8)
        self.avr_pool = nn.AvgPool2d(2)
    def forward(self,x):
        x = self.avr_pool(x)
        print("11111111", x.shape)
        
        
        
        x = F.interpolate(x , scale_factor = 2)
        print("2222222", x.shape)
        # x = x.permute(0,2,3,1).contiguous()
        # b,h,w,c = x.shape
        # w_size = self.window_size
        # print("111111111", x.shape)
        # x_inp = x.view(x.shape[0]*x.shape[1]//w_size[0]*x.shape[2]//w_size[1],
        #                w_size[0]*w_size[1],
        #                x.shape[3])
        # # x_inp = x_inp.permute(0,3,1,2).contiguous()
        
        # print("222222222", x_inp.shape)
        
        
        return x

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,size, factor ,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.factor = factor
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
        
        self.layer_norm_in = nn.LayerNorm(size)
        self.layer_norm_out = nn.LayerNorm(size)
        
        self.avr_pool = nn.AvgPool2d(self.factor)
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        
        x_in = x
        x = self.avr_pool(x)
        x = self.layer_norm_in(x)
        m_batchsize,C,width ,height = x.size()
        
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        # print('energy : ',energy.shape)
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        # print("111111", out.shape)
        out = out.view(m_batchsize,C,width,height)
        out = self.layer_norm_out(out)
        
        out = F.interpolate(out , scale_factor = self.factor)
        
        out = self.gamma*out + x_in
        return out

class scaled_ResnetGenerator(nn.Module):
    def __init__(self, args, norm_layer, n_blocks=4):
        super(scaled_ResnetGenerator, self).__init__()
        
        self.netG1 = ResnetGenerator(args.input_nc, 
                       args.output_nc, 
                       args.ngf, 
                       norm_layer=norm_layer, use_dropout=args.use_dropout, 
                       n_blocks=4,
                       learn_residual = args.learn_residual)
        self.netG2 = ResnetGenerator(args.input_nc, 
                       args.output_nc, 
                       args.ngf, 
                       norm_layer=norm_layer, use_dropout=args.use_dropout, 
                       n_blocks=4,
                       learn_residual = args.learn_residual)
        self.netG3 = ResnetGenerator(args.input_nc, 
                       args.output_nc, 
                       args.ngf, 
                       norm_layer=norm_layer, use_dropout=args.use_dropout, 
                       n_blocks=4,
                       learn_residual = args.learn_residual)
        
        self.tconv1 = nn.ConvTranspose2d(3, 3, kernel_size=5,stride=2, 
                                         padding=2,output_padding=1)
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5, stride=1,
                                       padding=2,padding_mode= 'reflect')
    def forward(self, x):
        x_scaled1 = torch.nn.functional.interpolate(x, (128, 128))
        x_scaled2 = torch.nn.functional.interpolate(x, (64, 64))
        x_scaled3 = torch.nn.functional.interpolate(x, (32, 32))
        
        # out = torch.cat((x_scaled3, x_scaled3), dim = 1)
        # out = self.netG1(out)
        # out = torch.nn.functional.interpolate(out, (64, 64))
        
        out = torch.cat((x_scaled1, x_scaled1), dim = 1)
        out = self.netG2(out)
        out = torch.nn.functional.interpolate(out, (256, 256))
        
        out = torch.cat((x, out), dim = 1)
        out = self.netG3(out)
        return out
        

class ResnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc,  ngf=64, device = None, norm_layer=nn.BatchNorm2d, 
            use_dropout=False,n_blocks=4, learn_residual=True,  padding_type='reflect',
            use_norm = True):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.learn_residual = learn_residual
        self.n_blocks = n_blocks
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        
        print('Norm_layer : ',  norm_layer)
        
        self.device = device
        self.layer1_ReflectionPad2d = nn.ReflectionPad2d(3)
        #################
        self.conv1 = nn.Conv2d(2*input_nc, 32, kernel_size=5, stride=1,
                                       padding=2,bias=use_bias, 
                                       padding_mode= 'reflect',dilation = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2,
                                       padding=2,bias=use_bias, 
                                       padding_mode= 'reflect',dilation = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2,
                                       padding=2,bias=use_bias,
                                       padding_mode= 'reflect',dilation = 1)
        self.tconv1 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, 
                                                output_padding=1, bias=use_bias)
        
        self.tconv2 = nn.ConvTranspose2d(64, 32, kernel_size=5,stride=2, 
                                         padding=2,output_padding=1, bias=use_bias)
        
        self.conv4 =nn.Conv2d(32, 3, kernel_size=5, stride=1,
                                       padding=2,bias=use_bias, 
                                       padding_mode= 'reflect',dilation = 1)
        #################
        
        self.diconv128 = dilated_conv_block(128, use_norm = use_norm, 
                                            norm_layer=norm_layer, use_bias=use_bias)
        self.diconv64 = dilated_conv_block(64, use_norm = use_norm, 
                                           norm_layer=norm_layer,use_bias=use_bias)
        self.diconv32 = dilated_conv_block(32, use_norm = use_norm, 
                                           norm_layer=norm_layer,use_bias=use_bias)
        ################
        self.BLOCK_256 = nn.ModuleList()
        for i in range(n_blocks):
            self.BLOCK_256.append(ResnetBlock(256, padding_type=padding_type, 
                                              use_norm = use_norm,
                                              norm_layer=norm_layer, 
                                              use_dropout=use_dropout, 
                                              use_bias=use_bias))
        ################
        BLOCK_128 = nn.ModuleList()
        for i in range(n_blocks):
            BLOCK_128.append(ResnetBlock(128, padding_type=padding_type, 
                                              use_norm = use_norm,
                                              norm_layer=norm_layer, 
                                              use_dropout=use_dropout, 
                                              use_bias=use_bias))
        self.BLOCK_128 = nn.Sequential(*BLOCK_128)
        ################
        BLOCK_64 = nn.ModuleList()
        for i in range(n_blocks):
            BLOCK_64.append(ResnetBlock(64, padding_type=padding_type, 
                                              use_norm = use_norm,
                                              norm_layer=norm_layer, 
                                              use_dropout=use_dropout, 
                                              use_bias=use_bias))
        self.BLOCK_64 = nn.Sequential(*BLOCK_64)
        ################
        BLOCK_32 = nn.ModuleList()
        for i in range(n_blocks):
            BLOCK_32.append(ResnetBlock(32, padding_type=padding_type, 
                                              use_norm = use_norm,
                                              norm_layer=norm_layer, 
                                              use_dropout=use_dropout, 
                                              use_bias=use_bias))
        self.BLOCK_32 = nn.Sequential(*BLOCK_32)
        ################
        BLOCK_64t = nn.ModuleList()
        for i in range(n_blocks):
            BLOCK_64t.append(ResnetBlock(64, padding_type=padding_type, 
                                              use_norm = use_norm,
                                              norm_layer=norm_layer, 
                                              use_dropout=use_dropout, 
                                              use_bias=use_bias))
        self.BLOCK_64t = nn.Sequential(*BLOCK_64t)
        #################
        BLOCK_32t = nn.ModuleList()
        for i in range(n_blocks):
            BLOCK_32t.append(ResnetBlock(32, padding_type=padding_type, 
                                              use_norm = use_norm,
                                              norm_layer=norm_layer, 
                                              use_dropout=use_dropout, 
                                              use_bias=use_bias))
        self.BLOCK_32t = nn.Sequential(*BLOCK_32t)
        #################
        
        self.layer4_ReflectionPad2d = nn.ReflectionPad2d(3)
        self.layer4_Conv2d = nn.Conv2d(64, output_nc, kernel_size=7, padding=0)
        self.layer4_Tanh = nn.Tanh()
        ################
        # self.attn1 = Self_Attn( 128, (128, 128,128), 1 , 'relu')
        # self.attn2 = Self_Attn( 256, (256, 64,64), 1 , 'relu')
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.diconv32(out)
        out = self.BLOCK_32(out)
        ##########
        out = self.conv2(out)
        out = self.diconv64(out)
        out = self.BLOCK_64(out)
        ##########
        out = self.conv3(out)
        out = self.BLOCK_128(out)
        ##########
        out = self.tconv1(out)
        out = self.BLOCK_64t(out)
        ##########
        out = self.tconv2(out)
        out = self.BLOCK_32t(out)
        
        out = self.conv4(out)
        out = torch.clamp( x[:,:3,:,:] + out, min=-1, max=1)
        
        return out

class dilated_conv_block(nn.Module):
    def __init__(self, dim, use_norm, norm_layer, use_bias):
        super(dilated_conv_block, self).__init__()
        
        self.dilated1_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2 , stride = 1,
                                      dilation = 1 ,bias=use_bias,padding_mode= 'reflect')
        self.dilated2_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=4 , stride = 1,
                                      dilation = 2,bias=use_bias,padding_mode= 'reflect')
        self.dilated3_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=6 , stride = 1,
                                      dilation = 3,bias=use_bias,padding_mode= 'reflect')
        self.dilated4_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=8 , stride = 1,
                                      dilation = 4,bias=use_bias,padding_mode= 'reflect')
        self.dilated5_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=10 , stride = 1,
                                      dilation = 5,bias=use_bias,padding_mode= 'reflect')
        
        self.conv = nn.Conv2d(5*dim, dim, kernel_size=5, padding=2 , stride = 1,
                              bias=use_bias)
        self.activation = nn.ReLU()
        self.norm_layer = norm_layer(dim)
        self.use_norm = use_norm
    def forward(self, x):
        x1 = self.dilated1_conv(x)
        x2 = self.dilated2_conv(x)
        x3 = self.dilated3_conv(x)
        x4 = self.dilated4_conv(x)
        x5 = self.dilated5_conv(x)
        x  = torch.cat((x1, x2, x3, x4, x5), dim = 1)
        x = self.conv(x)
        if self.use_norm:
            x = self.norm_layer(x)
        x = self.activation(x)
        return x
    
class ResnetBlock(nn.Module):

	def __init__(self, dim, padding_type, use_norm, norm_layer ,use_dropout, use_bias):
		super(ResnetBlock, self).__init__()

		padAndConv = {
			'reflect': [
                nn.ReflectionPad2d(2),
                nn.Conv2d(dim, dim, kernel_size=5, bias=use_bias)],
			'replicate': [
                nn.ReplicationPad2d(2),
                nn.Conv2d(dim, dim, kernel_size=5, bias=use_bias)],
			'zero': [
                nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=use_bias)]
		}

		try:
			blocks = padAndConv[padding_type] + \
                [norm_layer(dim)] if use_norm else [] + \
                    [nn.ReLU(True)] + \
                        [nn.Dropout(0.5)] if use_dropout else [] + \
                            padAndConv[padding_type] +\
                                [norm_layer(dim)] if use_norm else [] 
		except:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)

		self.conv_block = nn.Sequential(*blocks)

	def forward(self, x):
		out = x + self.conv_block(x)
		return out


    
    
class NLayerDiscriminator(nn.Module):
    def __init__(self, args, input_nc, norm_layer = nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        
        ndf=64
        n_layers=3
        device = None
        
        print('11111111111111',input_nc)
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        
        
        self.layer1 = nn.Conv2d(input_nc, ndf, kernel_size=kw, 
                                stride=2, padding=padw, dilation= 1,
                                padding_mode= 'reflect')
        self.relu1 = nn.LeakyReLU(0.2)
        
        self.layer2 = nn.Conv2d(input_nc, ndf, kernel_size=kw, 
                                stride=2, padding=padw+2, dilation= 2,
                                padding_mode= 'reflect')
        self.relu2 = nn.LeakyReLU(0.2)
        
        self.layer3 = nn.Conv2d(input_nc, ndf, kernel_size=kw, 
                                stride=2, padding=padw+5, dilation= 4,
                                padding_mode= 'reflect')
        self.relu3 = nn.LeakyReLU(0.2)
        
        
        # self.sequence1 = [
        #     nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
        #     nn.LeakyReLU(0.2, True)
        # ]
        
        self.layer5 = nn.Conv2d(3*ndf, ndf, kernel_size=3, 
                                stride=1, padding=padw)
        self.relu5 = nn.LeakyReLU(0.2)
        
        sequence =[]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, 
                      stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        # if use_sigmoid:
        #     sequence += [nn.Sigmoid()]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        
        out1 = self.layer1(input)
        out1 = self.relu1(out1)
        
        out2 = self.layer2(input)
        out2 = self.relu2(out2)
        
        out3 = self.layer3(input)
        out3 = self.relu3(out3)

        
        out = torch.cat((out1, out2, out3), dim = 1)
        
        
        out = self.layer5(out)
        out = self.relu5(out)
        
        return self.model(out)
    