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
from torch.utils.data import DataLoader

import cv2
import scipy.misc
import os
from skimage import io
from os import listdir
from os.path import join
from PIL import Image
import time
import glob

dtype = torch.cuda.FloatTensor

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

        y = torch.Tensor(x.size())
        y.fill_(0)
        y[:,:,:,1:] += x_diffx
        y[:,:,:,:-1] += x_diffx
        y[:,:,1:,:] += x_diffy
        y[:,:,:-1,:] += x_diffy
        y = torch.sum(y,1)/3
        y /= 4
        return y


def var_custom_collate(batch):
    min_h, min_w = 10000, 10000
    for item in batch:
        min_h = min(min_h, item['input_height'])
        min_w = min(min_w, item['input_width'])

    batch_input_images = torch.Tensor(len(batch), 4, min_h, min_w)
    batch_target_images = torch.Tensor(len(batch), 3, min_h, min_w)
    batch_input_params = torch.Tensor(len(batch), 2)
    for idx, item in enumerate(batch):
        off_y = random.randint(0, item['input_height']-min_h)
        off_x = random.randint(0, item['input_width']-min_w)
        batch_input_images[idx,0:3] = item['input_img'][:, :, off_y:off_y+min_h, off_x:off_x+min_w]
        batch_input_images[idx,3] = edgeCompute(batch_input_images.narrow(0,idx,1).narrow(1,0,3))
        batch_target_images[idx] = item['target_img'][:, :, off_y:off_y+min_h, off_x:off_x+min_w]
        batch_input_params[idx, 0] = item['input_param']
        batch_input_params[idx, 1] = item['input_type']

    return (batch_input_images, batch_input_params, batch_target_images)


class ArbitraryImageFolder(data.Dataset):

    def __init__(self, input_paths, label_paths, params, type, hws):
        self.input_paths = input_paths
        self.label_paths = label_paths
        self.params = params
        self.type= type
        self.hws = hws
        self.size = len(input_paths)

    def __getitem__(self, index):
        input_path, label_path, para, type = self.input_paths[index], self.label_paths[index], self.params[index], self.type[index]
        img_h, img_w = self.hws[index]

        input = cv2.imread(input_path)
        label = cv2.imread(label_path)

        input = input.transpose((2, 0, 1))
        input = torch.from_numpy(input).unsqueeze(0).float()
        label = label.transpose((2, 0, 1))
        label = torch.from_numpy(label).unsqueeze(0).float()

        return {'input_img': input, 'target_img': label, 'input_param': para, 'input_type': type, 'input_height': img_h, 'input_width': img_w}

    def __len__(self):
        return self.size





name = 'train_10_operator_model_realtime'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
epoches = 130
batchsize = 8
num_workers = 8

edgeCompute = EdgeComputation()

with open('/mnt/codes/imageOperator_codes/train_10_operator_model_realtime.py', 'r') as fin:
    print(fin.read())
print("***********************************************")

filter_input_folder = '/mnt/data/VOC2012_L0smooth_input/'

train_input_paths = []
train_label_paths = []
train_params = []
train_type = []
train_hws = []
with open("/mnt/data/VOC2012_10_operator_realtime_training_list.txt","r") as f:
    for line in f:
        label_path = line.strip()
        label_imgname = os.path.splitext(os.path.basename(label_path))[0]
        name_parts = label_imgname.split('_')
        img_h, img_w = int(name_parts[-2]), int(name_parts[-1])        
        type = label_path.split('/')[3].split('_')[1]
        if type == 'L0smooth':
            param = float(name_parts[-3])
            train_type.append(float('0.1'))
            train_params.append(param)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            train_label_paths.append(label_path)
        elif type == 'WLS':
            param = float(name_parts[-3])
            train_type.append(float('0.2'))
            train_params.append(param/50)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            train_label_paths.append(label_path)
        elif type == 'RTV':
            param = float(name_parts[-3])
            train_type.append(float('0.3'))
            train_params.append((param-0.002)*4.125+0.002)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            train_label_paths.append(label_path)
        elif type == 'fastLLFenhancement':
            param = float(name_parts[-3])
            train_type.append(float('0.4'))
            train_params.append((param-2)*0.033+0.002)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            train_label_paths.append(label_path)
        elif type == 'fastLLFenhancementgeneral':
            train_type.append(float('0.5'))
            train_params.append(0.02)
            base_imgname = label_imgname.replace('_%s_%s'%(name_parts[-2], name_parts[-1]), '')
            train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            train_label_paths.append(label_path)
        elif type == 'WLSenhancement':
            train_type.append(float('0.6'))
            train_params.append(0.02)
            base_imgname = label_imgname.replace('_%s_%s'%(name_parts[-2], name_parts[-1]), '')
            train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            train_label_paths.append(label_path)
        elif type == 'style':
            train_type.append(float('0.7'))
            train_params.append(0.02)
            base_imgname = label_imgname.replace('_%s_%s'%(name_parts[-2], name_parts[-1]), '')
            train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            train_label_paths.append(label_path)
        elif type == 'pencil':
            train_type.append(float('0.8'))
            train_params.append(0.02)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            train_label_paths.append(label_path)
        elif type == 'RGF':
            param = float(name_parts[-3])
            train_type.append(float('0.9'))
            train_params.append((param-1)*0.022+0.002)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            train_label_paths.append(label_path)
        elif type == 'WMF':
            param = float(name_parts[-3])
            train_type.append(float('1.0'))
            train_params.append((param-1)*0.022+0.002)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            train_label_paths.append(label_path)
        else:
            continue
        
        train_hws.append((img_h, img_w))

test_input_paths = []
test_label_paths = []
test_params = []
test_type = []
test_hws = []
with open("/mnt/data/VOC2012_10_operator_realtime_testing_list.txt","r") as f:
    for line in f:
        label_path = line.strip()
        label_imgname = os.path.splitext(os.path.basename(label_path))[0]
        name_parts = label_imgname.split('_')
        img_h, img_w = int(name_parts[-2]), int(name_parts[-1])
        type = (label_path.split('/')[3].split('_')[1])
        if type == 'L0smooth':
            param = float(name_parts[-3])
            test_type.append(float('0.1'))
            test_params.append(param)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            test_label_paths.append(label_path)
        elif type == 'WLS':
            param = float(name_parts[-3])
            test_type.append(float('0.2'))
            test_params.append(param/50)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            test_label_paths.append(label_path)
        elif type == 'RTV':
            param = float(name_parts[-3])
            test_type.append(float('0.3'))
            test_params.append((param-0.002)*4.125+0.002)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            test_label_paths.append(label_path)
        elif type == 'fastLLFenhancement':
            param = float(name_parts[-3])
            test_type.append(float('0.4'))
            test_params.append((param-2)*0.033+0.002)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            test_label_paths.append(label_path)
        elif type == 'fastLLFenhancementgeneral':
            test_type.append(float('0.5'))
            test_params.append(0.02)
            base_imgname = label_imgname.replace('_%s_%s'%(name_parts[-2], name_parts[-1]), '')
            test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            test_label_paths.append(label_path)
        elif type == 'WLSenhancement':
            test_type.append(float('0.6'))
            test_params.append(0.02)
            base_imgname = label_imgname.replace('_%s_%s'%(name_parts[-2], name_parts[-1]), '')
            test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            test_label_paths.append(label_path)
        elif type == 'style':
            test_type.append(float('0.7'))
            test_params.append(0.02)
            base_imgname = label_imgname.replace('_%s_%s'%(name_parts[-2], name_parts[-1]), '')
            test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            test_label_paths.append(label_path)
        elif type == 'pencil':
            test_type.append(float('0.8'))
            test_params.append(0.02)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            test_label_paths.append(label_path)
        elif type == 'RGF':
            param = float(name_parts[-3])
            test_type.append(float('0.9'))
            test_params.append((param-1)*0.022+0.002)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            test_label_paths.append(label_path)
        elif type == 'WMF':
            param = float(name_parts[-3])
            test_type.append(float('1.0'))
            test_params.append((param-1)*0.022+0.002)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
            test_label_paths.append(label_path)
        else:
            continue

        test_hws.append((img_h, img_w))

trainSetSize, testSetSize = len(train_input_paths), len(test_input_paths)
epochSize = 15983

model = Model()
criterion = torch.nn.MSELoss(size_average=True)
model = torch.nn.DataParallel(model.cuda())
criterion = criterion.cuda()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_dataset = ArbitraryImageFolder(train_input_paths, train_label_paths, train_params, train_type, train_hws)
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=var_custom_collate,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

test_dataset = ArbitraryImageFolder(test_input_paths, test_label_paths, test_params, test_type, test_hws)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, collate_fn=var_custom_collate,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

total_start_time = time.time()
iter = 0
iterTest = 0
for p in range(epoches):
    train_loss_all = [0,0,0,0,0,0,0,0,0,0]
    test_loss_all = [0,0,0,0,0,0,0,0,0,0]
    train_count_all = [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    test_count_all = [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    train_loss = 0
    test_loss = 0
    train_count = 0
    test_count = 0
    start_time = time.time()
    if p == 90:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if p == 110:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    if p == 120:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)


    for curr_iter, batch_data in enumerate(train_dataloader):
        iter += 1

        inputTensor, inputParaTensor, labelTensor = batch_data
        inputTensor = inputTensor - 128
        inputTensor_v = Variable(inputTensor.cuda())
        inputParaTensor_v = Variable(inputParaTensor.cuda())
        labelTensor_v = Variable(labelTensor.cuda())

        optimizer.zero_grad()
        pred_v = model([inputParaTensor_v, inputTensor_v])
        loss = criterion(pred_v, labelTensor_v)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_count += 1
        print('Iter: %d Current loss: %.4f'%(iter,loss.item()))

        for n in range(batchsize):
            index = int(round(inputParaTensor[n,1].data[0].cpu().numpy() * 10 - 1))
            temp_loss = criterion(pred_v[n], labelTensor_v[n])
            train_loss_all[index] += temp_loss.item()
            train_count_all[index] += 1 

        if iter % int(epochSize/batchsize) == 0:
            break

    for curr_iter, batch_data in enumerate(test_dataloader):
        iterTest += 1

        inputTensor, inputParaTensor, labelTensor = batch_data
        inputTensor = inputTensor - 128
        inputTensor_v = Variable(inputTensor.cuda())
        inputParaTensor_v = Variable(inputParaTensor.cuda())
        labelTensor_v = Variable(labelTensor.cuda())

        pred_v = model([inputParaTensor_v, inputTensor_v])
        loss = criterion(pred_v, labelTensor_v)
        loss.backward()

        test_loss += loss.item()
        test_count += 1

        for n in range(batchsize):
            index = int(round(inputParaTensor[n,1].data[0].cpu().numpy() * 10 - 1))
            temp_loss = criterion(pred_v[n], labelTensor_v[n])
            test_loss_all[index] += temp_loss.item()
            test_count_all[index] += 1 

        if iterTest % int(epochSize/(batchsize*20)) == 0:
            break

    train_loss = train_loss / train_count
    test_loss = test_loss / test_count
    elapsed = time.time() - start_time
    print('Epoch: %d train loss: L0smooth: %.4f, WLS: %.4f, RTV: %.4f, fastLLFenhancement: %.4f, fastLLFenhancementgeneral: %.4f, WLSenhancement: %.4f, style: %.4f, pencil: %.4f, RGF: %.4f, WMF: %.4f'%(p+1,train_loss_all[0]/train_count_all[0],train_loss_all[1]/train_count_all[1],train_loss_all[2]/train_count_all[2],train_loss_all[3]/train_count_all[3],train_loss_all[4]/train_count_all[4],train_loss_all[5]/train_count_all[5],train_loss_all[6]/train_count_all[6],train_loss_all[7]/train_count_all[7],train_loss_all[8]/train_count_all[8],train_loss_all[9]/train_count_all[9]))
    print('Epoch: %d train loss: L0smooth: %.4f, WLS: %.4f, RTV: %.4f, fastLLFenhancement: %.4f, fastLLFenhancementgeneral: %.4f, WLSenhancement: %.4f, style: %.4f, pencil: %.4f, RGF: %.4f, WMF: %.4f'%(p+1,test_loss_all[0]/test_count_all[0],test_loss_all[1]/test_count_all[1],test_loss_all[2]/test_count_all[2],test_loss_all[3]/test_count_all[3],test_loss_all[4]/test_count_all[4],test_loss_all[5]/test_count_all[5],test_loss_all[6]/test_count_all[6],test_loss_all[7]/test_count_all[7],test_loss_all[8]/test_count_all[8],test_loss_all[9]/test_count_all[9]))
    print('Epoch: %d average train loss: %.4f'%(p+1,train_loss))
    print('Epoch: %d average test  loss: %.4f'%(p+1,test_loss))
    print('Epoch: %d time elapsed: %.2f hours'%(p+1,elapsed/3600))
    torch.save(model.module.state_dict(),'/mnt/codes/imageOperator/netfiles/model_{}_{}.net'.format(name,p+1))
total_elapsed = time.time() - total_start_time
print('Total time elapsed: %.2f days'%(total_elapsed/(3600*24)))