import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.data as data
import math

from torch.backends import cudnn
from torch.autograd import Variable

import cv2
import skimage
from skimage import io
from os import listdir
from os.path import join
from PIL import Image
import os
import time
import glob, os


def save_image(filename, data):
    img = data.numpy()
    img = np.clip(img,0,255)
    img = img.transpose(1, 2, 0).astype("uint8")

    cv2.imwrite(filename,img)

class Resize(nn.Module):
    def __init__(self, a,b,c,d):
        super(Resize, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def forward(self, x):
        x = x.resize(self.a,self.b,self.c,self.d)
        return x

class EdgeComputation(nn.Module):
    def __init__(self):
        super(EdgeComputation, self).__init__()

    def forward(self, x):
        x_diffx = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
        x_diffy = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])

        y = torch.Tensor(x.size()).cuda()
        y.fill_(0)
        y[:,:,:,1:] += x_diffx
        y[:,:,:,:-1] += x_diffx
        y[:,:,1:,:] += x_diffy
        y[:,:,:-1,:] += x_diffy
        y = torch.sum(y,1)/3
        y /= 4
        return y

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.relu = nn.ReLU()

        self.fc1_weight = nn.Linear(2,16)
        self.fc1_bias = nn.Linear(2,16)
        self.fc2_weight = nn.Linear(16,64)
        self.fc2_bias = nn.Linear(16,64)

        self.conv1_weight = nn.Parameter(torch.Tensor(128,4,3,3).normal_(0, 1))
        self.conv2_weight = nn.Parameter(torch.Tensor(128,128,3,3).normal_(0, 1))
        self.conv3_weight = nn.Parameter(torch.Tensor(128,128,3,3).normal_(0, 1))

        self.conv4_weight = nn.Parameter(torch.Tensor(256,128,1,1).normal_(0, 1))
        self.conv5_weight = nn.Parameter(torch.Tensor(256,1,3,3).normal_(0, 1))
        self.conv6_weight = nn.Parameter(torch.Tensor(128,256,1,1).normal_(0, 1))

        self.conv7_weight = nn.Parameter(torch.Tensor(256,128,1,1).normal_(0, 1))
        self.conv8_weight = nn.Parameter(torch.Tensor(256,1,3,3).normal_(0, 1))
        self.conv9_weight = nn.Parameter(torch.Tensor(128,256,1,1).normal_(0, 1))

        self.conv10_weight = nn.Parameter(torch.Tensor(256,128,1,1).normal_(0, 1))
        self.conv11_weight = nn.Parameter(torch.Tensor(256,1,3,3).normal_(0, 1))
        self.conv12_weight = nn.Parameter(torch.Tensor(128,256,1,1).normal_(0, 1))

        self.conv13_weight = nn.Parameter(torch.Tensor(256,128,1,1).normal_(0, 1))
        self.conv14_weight = nn.Parameter(torch.Tensor(256,1,3,3).normal_(0, 1))
        self.conv15_weight = nn.Parameter(torch.Tensor(128,256,1,1).normal_(0, 1))

        self.conv16_weight = nn.Parameter(torch.Tensor(256,128,1,1).normal_(0, 1))
        self.conv17_weight = nn.Parameter(torch.Tensor(256,1,3,3).normal_(0, 1))
        self.conv18_weight = nn.Parameter(torch.Tensor(128,256,1,1).normal_(0, 1))

        self.conv19_weight = nn.Parameter(torch.Tensor(256,128,1,1).normal_(0, 1))
        self.conv20_weight = nn.Parameter(torch.Tensor(256,1,3,3).normal_(0, 1))
        self.conv21_weight = nn.Parameter(torch.Tensor(128,256,1,1).normal_(0, 1))

        self.conv22_weight = nn.Parameter(torch.Tensor(256,128,1,1).normal_(0, 1))
        self.conv23_weight = nn.Parameter(torch.Tensor(256,1,3,3).normal_(0, 1))
        self.conv24_weight = nn.Parameter(torch.Tensor(128,256,1,1).normal_(0, 1))

        self.conv25_weight = nn.Parameter(torch.Tensor(128,128,4,4).normal_(0, 1))
        self.conv26_weight = nn.Parameter(torch.Tensor(64,128,3,3).normal_(0, 1))
        self.conv27_weight = nn.Parameter(torch.Tensor(3,64,3,3).normal_(0, 1))
        self.conv27_bias = nn.Parameter(torch.Tensor(3).normal_(0, 1))

        self.norm_1 = nn.InstanceNorm2d(128, affine=True)
        self.norm_2 = nn.InstanceNorm2d(128, affine=True)
        self.norm_3 = nn.InstanceNorm2d(128, affine=True)

        self.norm_4 = nn.InstanceNorm2d(256, affine=True)
        self.norm_5 = nn.InstanceNorm2d(256, affine=True)
        self.norm_6 = nn.InstanceNorm2d(128, affine=True)
        
        self.norm_7 = nn.InstanceNorm2d(256, affine=True)
        self.norm_8 = nn.InstanceNorm2d(256, affine=True)
        self.norm_9 = nn.InstanceNorm2d(128, affine=True)
        
        self.norm_10 = nn.InstanceNorm2d(256, affine=True)
        self.norm_11 = nn.InstanceNorm2d(256, affine=True)
        self.norm_12 = nn.InstanceNorm2d(128, affine=True)
        
        self.norm_13 = nn.InstanceNorm2d(256, affine=True)
        self.norm_14 = nn.InstanceNorm2d(256, affine=True)
        self.norm_15 = nn.InstanceNorm2d(128, affine=True)
        
        self.norm_16 = nn.InstanceNorm2d(256, affine=True)
        self.norm_17 = nn.InstanceNorm2d(256, affine=True)
        self.norm_18 = nn.InstanceNorm2d(128, affine=True)
        
        self.norm_19 = nn.InstanceNorm2d(256, affine=True)
        self.norm_20 = nn.InstanceNorm2d(256, affine=True)
        self.norm_21 = nn.InstanceNorm2d(128, affine=True)
        
        self.norm_22 = nn.InstanceNorm2d(256, affine=True)
        self.norm_23 = nn.InstanceNorm2d(256, affine=True)
        self.norm_24 = nn.InstanceNorm2d(128, affine=True)
        
        self.norm_25 = nn.InstanceNorm2d(128, affine=True)
        self.norm_26 = nn.InstanceNorm2d(128, affine=True)

        for m in self.modules():
            if isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        x_param = input[0]
        x_img = input[1]

        norm_weight = self.fc2_weight(self.relu(self.fc1_weight(x_param)))
        norm_bias = self.fc2_bias(self.relu(self.fc1_bias(x_param)))

        input_img = x_img.narrow(1,0,3) + 128

        conv1_feat = self.relu(self.norm_1(
            F.conv2d(x_img, self.conv1_weight, bias=None, padding=1)))
        conv2_feat = self.relu(self.norm_2(
            F.conv2d(conv1_feat, self.conv2_weight, bias=None, padding=1)))
        conv3_feat = self.relu(self.norm_3(
            F.conv2d(conv2_feat, self.conv3_weight, bias=None, stride=2, padding=1)))

        # residual block
        conv4_feat = self.relu(self.norm_4(
            F.conv2d(conv3_feat, self.conv4_weight, bias=None, stride=1, padding=0, dilation=2)))
        conv5_feat = self.relu(self.norm_5(
            F.conv2d(conv4_feat, self.conv5_weight, bias=None, stride=1, padding=2, dilation=2, groups = 256)))
        conv6_feat = (self.norm_6(
            F.conv2d(conv5_feat, self.conv6_weight, bias=None, stride=1, padding=0, dilation=2)))
        conv6_feat = self.relu(conv3_feat + conv6_feat)

        # residual block
        conv7_feat = self.relu(self.norm_7(
            F.conv2d(conv6_feat, self.conv7_weight, bias=None, stride=1, padding=0, dilation=4)))
        conv8_feat = self.relu(self.norm_8(
            F.conv2d(conv7_feat, self.conv8_weight, bias=None, stride=1, padding=4, dilation=4, groups = 256)))
        conv9_feat = self.norm_9(
            F.conv2d(conv8_feat, self.conv9_weight, bias=None, stride=1, padding=0, dilation=4))
        conv9_feat = self.relu(conv6_feat + conv9_feat)

        # residual block
        conv10_feat = self.relu(self.norm_10(
            F.conv2d(conv9_feat, self.conv10_weight, bias=None, stride=1, padding=0, dilation=4)))
        conv11_feat = self.relu(self.norm_11(
            F.conv2d(conv10_feat, self.conv11_weight, bias=None, stride=1, padding=4, dilation=4, groups = 256)))
        conv12_feat = (self.norm_12(
            F.conv2d(conv11_feat, self.conv12_weight, bias=None, stride=1, padding=0, dilation=4)))
        conv12_feat = self.relu(conv9_feat + conv12_feat)

        # residual block
        conv13_feat = self.relu(self.norm_13(
            F.conv2d(conv12_feat, self.conv13_weight, bias=None, stride=1, padding=0, dilation=8)))
        conv14_feat = self.relu(self.norm_14(
            F.conv2d(conv13_feat, self.conv14_weight, bias=None, stride=1, padding=8, dilation=8, groups = 256)))
        conv15_feat = self.norm_15(
            F.conv2d(conv14_feat, self.conv15_weight, bias=None, stride=1, padding=0, dilation=8))
        conv15_feat = self.relu(conv12_feat + conv15_feat)

        # residual block
        conv16_feat = self.relu(self.norm_16(
            F.conv2d(conv15_feat, self.conv16_weight, bias=None, stride=1, padding=0, dilation=8)))
        conv17_feat = self.relu(self.norm_17(
            F.conv2d(conv16_feat, self.conv17_weight, bias=None, stride=1, padding=8, dilation=8, groups = 256)))
        conv18_feat = self.norm_18(
            F.conv2d(conv17_feat, self.conv18_weight, bias=None, stride=1, padding=0, dilation=8))
        conv18_feat = self.relu(conv15_feat + conv18_feat)

        # residual block
        conv19_feat = self.relu(self.norm_19(
            F.conv2d(conv18_feat, self.conv19_weight, bias=None, stride=1, padding=0, dilation=16)))
        conv20_feat = self.relu(self.norm_20(
            F.conv2d(conv19_feat, self.conv20_weight, bias=None, stride=1, padding=16, dilation=16, groups = 256)))
        conv21_feat = self.norm_21(
            F.conv2d(conv20_feat, self.conv21_weight, bias=None, stride=1, padding=0, dilation=16))
        conv21_feat = self.relu(conv18_feat + conv21_feat)

        # residual block
        conv22_feat = self.relu(self.norm_22(
            F.conv2d(conv21_feat, self.conv22_weight, bias=None, stride=1, padding=0, dilation=1)))
        conv23_feat = self.relu(self.norm_23(
            F.conv2d(conv22_feat, self.conv23_weight, bias=None, stride=1, padding=1, dilation=1, groups = 256)))
        conv24_feat = self.norm_24(
            F.conv2d(conv23_feat, self.conv24_weight, bias=None, stride=1, padding=0, dilation=1))
        conv24_feat = self.relu(conv21_feat + conv24_feat)

        # last 3 convs
        conv25_feat = self.relu(self.norm_25(
            F.conv_transpose2d(conv24_feat, self.conv25_weight, bias=None, stride=2, padding=1)))

        batch_size = x_img.size(0)
        batch_outputs = []
        for n in range(batch_size):
            conv26_feat_norm = F.conv2d(conv25_feat.narrow(0,n,1), self.conv26_weight, bias=None, stride=1, padding=1)
            conv26_feat = self.relu(F.instance_norm(conv26_feat_norm, weight = norm_weight[n], bias = norm_bias[n]))
            batch_outputs.append(conv26_feat)
        conv26_feat = torch.cat(batch_outputs, dim=0)

        output_residual = F.conv2d(conv26_feat, self.conv27_weight, bias=self.conv27_bias, stride=1, padding=1)

        output = input_img + output_residual

        return output





os.environ['CUDA_VISIBLE_DEVICES'] = '0'
netfile = './models/model_joint_10_operator_realtime.net'
savepath = './results/'

edgeCompute = EdgeComputation()

model = Model()
model.load_state_dict(torch.load(netfile))
model = model.cuda()
model.eval()

criterion = torch.nn.MSELoss(size_average=True)
criterion = criterion.cuda()

# note all the parameter values will be aligned to the numerical range [0.002, 0.2]
type = ['L0smooth','WLS','RTV','RGF','WMF','fastLLFenhancement','fastLLFenhancementgeneral','WLSenhancement','style','pencil']
totalPara = [
['000200', '000431', '002000', '009283', '020000'],
['010000', '021544', '100000', '464159', '1000000'],
['000200', '000447', '001000', '002236', '005000'],
['1.00000','3.25000','5.50000','7.75000','10.00000'],
['1.00000','3.25000','5.50000','7.75000','10.00000'],
['2','3','5','7','8'],
['1.00000'],
['1.00000'],
['1.00000'],
['1.00000']
]

for m in range(len(type)):
    para = totalPara[m]
    loss_mse_ave = 0
    loss_psnr_ave = 0
    print(type[m])
    for n in range(len(para)):
        loss_mse = 0
        loss_psnr = 0
        num = 0

        inputNameList, saveNameList, labelNameList = [], [], []
        if type[m] == 'L0smooth':
            print(float('%s.%s'%(para[n][:-5],para[n][-5:])))
            parameter = float('%s.%s'%(para[n][:-5],para[n][-5:]))
            input_para = torch.zeros(1,2)
            input_para[0,0] = parameter
            input_para[0,1] = 0.1
            inputPath = "./test_filter/"
            searchPath = "./test_filter/*-input.png"
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'input','L0smooth-%s'%(para[n]))
                saveName = str.replace(inputName,'input','L0smooth-%s-predict'%(para[n]))
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'WLS':
            print(float('%s.%s'%(para[n][:-5],para[n][-5:])))
            parameter = float('%s.%s'%(para[n][:-5],para[n][-5:]))
            input_para = torch.zeros(1,2)
            input_para[0,0] = parameter/50
            input_para[0,1] = 0.2
            inputPath = "./test_filter/"
            searchPath = "./test_filter/*-input.png"
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'input','WLS-%s'%(para[n]))
                saveName = str.replace(inputName,'input','WLS-%s-predict'%(para[n]))
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'RTV':
            print(float('%s.%s'%(para[n][:-5],para[n][-5:])))
            parameter = float('%s.%s'%(para[n][:-5],para[n][-5:]))
            input_para = torch.zeros(1,2)
            input_para[0,0] = (parameter-0.002)*4.125+0.002
            input_para[0,1] = 0.3
            inputPath = "./test_filter/"
            searchPath = "./test_filter/*-input.png"
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'input','RTV-%s'%(para[n]))
                saveName = str.replace(inputName,'input','RTV-%s-predict'%(para[n]))
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'fastLLFenhancement':
            print(float('%s'%(para[n])))
            parameter = float('%s'%(para[n]))
            input_para = torch.zeros(1,2)
            input_para[0,0] = (parameter-2)*0.033+0.002
            input_para[0,1] = 0.4
            inputPath = "./test_filter/"
            searchPath = "./test_filter/*-input.png"
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'input','fast_LLFenhancement-%s'%(para[n]))
                saveName = str.replace(inputName,'input','fast_LLFenhancement-%s-predict'%(para[n]))
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'fastLLFenhancementgeneral':
            input_para = torch.zeros(1,2)
            input_para[0,0] = 0.02
            input_para[0,1] = 0.5
            inputPath = "./test_filter/"
            searchPath = "./test_filter/*-input.png"
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'input','fast_LLFenhancementgeneral')
                saveName = str.replace(inputName,'input','fast_LLFenhancementgeneral-predict')
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'WLSenhancement':
            input_para = torch.zeros(1,2)
            input_para[0,0] = 0.02
            input_para[0,1] = 0.6
            inputPath = "./test_filter/"
            searchPath = "./test_filter/*-input.png"
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'input','WLSenhancement')
                saveName = str.replace(inputName,'input','WLSenhancement-predict')
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'style':
            input_para = torch.zeros(1,2)
            input_para[0,0] = 0.02
            input_para[0,1] = 0.7
            inputPath = "./test_filter/"
            searchPath = "./test_filter/*-input.png"
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'input','style')
                saveName = str.replace(inputName,'input','style-predict')
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'pencil':
            input_para = torch.zeros(1,2)
            input_para[0,0] = 0.02
            input_para[0,1] = 0.8
            inputPath = "./test_filter/"
            searchPath = "./test_filter/*-input.png"
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'input','pencilColor')
                saveName = str.replace(inputName,'input','pencilColor-predict')
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'RGF':
            print(float('%s'%(para[n])))
            parameter = float('%s'%(para[n]))
            input_para = torch.zeros(1,2)
            input_para[0,0] = (parameter-1)*0.022+0.002
            input_para[0,1] = 0.9
            inputPath = "./test_filter/"
            searchPath = "./test_filter/*-input.png"
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'input','RGF-%s'%(para[n]))
                saveName = str.replace(inputName,'input','RGF-%s-predict'%(para[n]))
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'WMF':
            print(float('%s'%(para[n])))
            parameter = float('%s'%(para[n]))
            input_para = torch.zeros(1,2)
            input_para[0,0] = (parameter-1)*0.022+0.002
            input_para[0,1] = 1.0
            inputPath = "./test_filter/"
            searchPath = "./test_filter/*-input.png"
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'input','WMF-%s'%(para[n]))
                saveName = str.replace(inputName,'input','WMF-%s-predict'%(para[n]))
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)


        for i in range(len(inputNameList)):
            inputName = inputNameList[i]
            labelName = labelNameList[i]
            saveName = saveNameList[i]

            saveName_input = str.replace(inputName,inputPath,savepath)
            saveName = str.replace(saveName,inputPath,savepath)
            saveName_label = str.replace(labelName,inputPath,savepath)

            input = cv2.imread(inputName)
            label = cv2.imread(labelName)

            input = input.transpose((2, 0, 1))
            label = label.transpose((2, 0, 1))
            input = torch.from_numpy(input)
            input = input.unsqueeze(0).float()
            label = torch.from_numpy(label)
            label = label.unsqueeze(0).float()
            input = input.cuda()
            
            inputs = torch.zeros(1,4,input.size(2),input.size(3))
            inputs[0,0:3] = input - 128
            inputs[0,3] = edgeCompute(input) - 128

            input_para = Variable(input_para)
            input_para = input_para.cuda()
            
            inputs = inputs.cuda()
            label = label.cuda()
            inputs = Variable(inputs)
            input = Variable(input)
            label = Variable(label, requires_grad=False)

            pred = model([input_para,inputs])
            loss = criterion(pred, label)
            loss.backward()

            # for j in range(3):
            #     numerator = torch.dot(pred[0,j].view(-1),input[0,j].view(-1))
            #     denominator = torch.dot(pred[0,j].view(-1),pred[0,j].view(-1))
            #     alpha = numerator/denominator
            #     pred[0,j] = pred[0,j] * alpha

            pred_numpy = pred.data[0].cpu().numpy()
            label_numpy = label.data[0].cpu().numpy()
            loss =  np.mean((pred_numpy - label_numpy)**2)
            loss_mse += loss
            loss_psnr += 10 *np.log10(255*255/loss)
            num += 1

            pred = pred.data[0].cpu()
            label = label.data[0].cpu()
            input = input.data[0].cpu()
            save_image(saveName,pred)
            save_image(saveName_label,label)
            save_image(saveName_input,input)

        print("mse: %f, psnr: %f"%(loss_mse/num,loss_psnr/num))
        loss_psnr_ave = loss_psnr_ave + loss_psnr/num
        loss_mse_ave = loss_mse_ave + loss_mse/num
    print("average mse: %f, psnr: %f"%(loss_mse_ave/len(para),loss_psnr_ave/len(para)))