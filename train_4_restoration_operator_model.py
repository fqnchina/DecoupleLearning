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
        self.resize1 = Resize(64,4,3,3)
        self.resize2 = Resize(64,64,3,3)
        self.resize3 = Resize(3,64,3,3)
        self.resize4 = Resize(64,64,4,4)

        self.fc1 = nn.Linear(1,2304)
        self.fc2 = nn.Linear(1,36864)
        self.fc3 = nn.Linear(1,36864)
        self.fc4 = nn.Linear(1,36864)
        self.fc5 = nn.Linear(1,36864)
        self.fc6 = nn.Linear(1,36864)
        self.fc7 = nn.Linear(1,36864)
        self.fc8 = nn.Linear(1,36864)
        self.fc9 = nn.Linear(1,36864)
        self.fc10 = nn.Linear(1,36864)
        self.fc11 = nn.Linear(1,36864)
        self.fc12 = nn.Linear(1,36864)
        self.fc13 = nn.Linear(1,36864)
        self.fc14 = nn.Linear(1,36864)
        self.fc15 = nn.Linear(1,36864)
        self.fc16 = nn.Linear(1,36864)
        self.fc17 = nn.Linear(1,36864)
        self.fc18 = nn.Linear(1,65536)
        self.fc19 = nn.Linear(1,36864)
        self.fc20 = nn.Linear(1,1728)
        
        self.norm_1 = nn.InstanceNorm2d(64, affine=True)
        self.norm_2 = nn.InstanceNorm2d(64, affine=True)
        self.norm_3 = nn.InstanceNorm2d(64, affine=True)
        self.norm_4 = nn.InstanceNorm2d(64, affine=True)
        self.norm_5 = nn.InstanceNorm2d(64, affine=True)
        self.norm_6 = nn.InstanceNorm2d(64, affine=True)
        self.norm_7 = nn.InstanceNorm2d(64, affine=True)
        self.norm_8 = nn.InstanceNorm2d(64, affine=True)
        self.norm_9 = nn.InstanceNorm2d(64, affine=True)
        self.norm_10 = nn.InstanceNorm2d(64, affine=True)
        self.norm_11 = nn.InstanceNorm2d(64, affine=True)
        self.norm_12 = nn.InstanceNorm2d(64, affine=True)
        self.norm_13 = nn.InstanceNorm2d(64, affine=True)
        self.norm_14 = nn.InstanceNorm2d(64, affine=True)
        self.norm_15 = nn.InstanceNorm2d(64, affine=True)
        self.norm_16 = nn.InstanceNorm2d(64, affine=True)
        self.norm_17 = nn.InstanceNorm2d(64, affine=True)
        self.norm_18 = nn.InstanceNorm2d(64, affine=True)
        self.norm_19 = nn.InstanceNorm2d(64, affine=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.ConvTranspose3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        x_param = input[0]
        x_img = input[1]

        conv1_weight = self.fc1(x_param)
        conv2_weight = self.fc2(x_param)
        conv3_weight = self.fc3(x_param)
        conv4_weight = self.fc4(x_param)
        conv5_weight = self.fc5(x_param)
        conv6_weight = self.fc6(x_param)
        conv7_weight = self.fc7(x_param)
        conv8_weight = self.fc8(x_param)
        conv9_weight = self.fc9(x_param)
        conv10_weight = self.fc10(x_param)
        conv11_weight = self.fc11(x_param)
        conv12_weight = self.fc12(x_param)
        conv13_weight = self.fc13(x_param)
        conv14_weight = self.fc14(x_param)
        conv15_weight = self.fc15(x_param)
        conv16_weight = self.fc16(x_param)
        conv17_weight = self.fc17(x_param)
        conv18_weight = self.fc18(x_param)
        conv19_weight = self.fc19(x_param)
        conv20_weight = self.fc20(x_param)

        batch_size = x_img.size(0)
        batch_outputs = []
        for n in range(batch_size):
            # first 3 convs
            conv1_feat = self.relu(self.norm_1(
                F.conv2d(x_img.narrow(0, n, 1), self.resize1(conv1_weight[n]), bias=None, padding=1)))
            conv2_feat = self.relu(self.norm_2(
                F.conv2d(conv1_feat, self.resize2(conv2_weight[n]), bias=None, padding=1)))
            conv3_feat = self.relu(self.norm_3(
                F.conv2d(conv2_feat,self.resize2(conv3_weight[n]), bias=None, stride=2, padding=1)))

            # residual block
            conv4_feat = self.relu(self.norm_4(
                F.conv2d(conv3_feat, self.resize2(conv4_weight[n]), bias=None, stride=1, padding=2, dilation=2)))
            conv5_feat = self.norm_5(
                F.conv2d(conv4_feat, self.resize2(conv5_weight[n]), bias=None, stride=1, padding=2, dilation=2))
            conv5_feat = self.relu(conv3_feat + conv5_feat)

            # residual block
            conv6_feat = self.relu(self.norm_6(
                F.conv2d(conv5_feat, self.resize2(conv6_weight[n]), bias=None, stride=1, padding=4, dilation=4)))
            conv7_feat = self.norm_7(
                F.conv2d(conv6_feat, self.resize2(conv7_weight[n]), bias=None, stride=1, padding=4, dilation=4))
            conv7_feat = self.relu(conv5_feat + conv7_feat)

            # residual block
            conv8_feat = self.relu(self.norm_8(
                F.conv2d(conv7_feat, self.resize2(conv8_weight[n]), bias=None, stride=1, padding=4, dilation=4)))
            conv9_feat = self.norm_9(
                F.conv2d(conv8_feat, self.resize2(conv9_weight[n]), bias=None, stride=1, padding=4, dilation=4))
            conv9_feat = self.relu(conv7_feat + conv9_feat)

            # residual block
            conv10_feat = self.relu(self.norm_10(
                F.conv2d(conv9_feat, self.resize2(conv10_weight[n]), bias=None, stride=1, padding=8, dilation=8)))
            conv11_feat = self.norm_11(
                F.conv2d(conv10_feat, self.resize2(conv11_weight[n]), bias=None, stride=1, padding=8, dilation=8))
            conv11_feat = self.relu(conv9_feat + conv11_feat)

            # residual block
            conv12_feat = self.relu(self.norm_12(
                F.conv2d(conv11_feat, self.resize2(conv12_weight[n]), bias=None, stride=1, padding=8, dilation=8)))
            conv13_feat = self.norm_13(
                F.conv2d(conv12_feat, self.resize2(conv13_weight[n]), bias=None, stride=1, padding=8, dilation=8))
            conv13_feat = self.relu(conv11_feat + conv13_feat)

            # residual block
            conv14_feat = self.relu(self.norm_14(
                F.conv2d(conv13_feat, self.resize2(conv14_weight[n]), bias=None, stride=1, padding=16, dilation=16)))
            conv15_feat = self.norm_15(
                F.conv2d(conv14_feat, self.resize2(conv15_weight[n]), bias=None, stride=1, padding=16, dilation=16))
            conv15_feat = self.relu(conv13_feat + conv15_feat)

            # residual block
            conv16_feat = self.relu(self.norm_16(
                F.conv2d(conv15_feat, self.resize2(conv16_weight[n]), bias=None, stride=1, padding=1, dilation=1)))
            conv17_feat = self.norm_17(
                F.conv2d(conv16_feat, self.resize2(conv17_weight[n]), bias=None, stride=1, padding=1, dilation=1))
            conv17_feat = self.relu(conv15_feat + conv17_feat)

            # last 3 convs
            conv18_feat = self.relu(self.norm_18(
                F.conv_transpose2d(conv17_feat, self.resize4(conv18_weight[n]), bias=None, stride=2, padding=1)))
            conv19_feat = self.relu(self.norm_19(
                F.conv2d(conv18_feat, self.resize2(conv19_weight[n]), bias=None, stride=1, padding=1)))
            output = self.relu(F.conv2d(conv19_feat, self.resize3(conv20_weight[n]), bias=None, stride=1, padding=1))

            batch_outputs.append(output)

        output = torch.cat(batch_outputs, dim=0)
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
    batch_input_params = torch.Tensor(len(batch), 1)
    for idx, item in enumerate(batch):
        off_y = random.randint(0, item['input_height']-min_h)
        off_x = random.randint(0, item['input_width']-min_w)
        batch_input_images[idx,0:3] = item['input_img'][:, :, off_y:off_y+min_h, off_x:off_x+min_w]
        batch_input_images[idx,3] = edgeCompute(batch_input_images.narrow(0,idx,1).narrow(1,0,3))
        batch_target_images[idx] = item['target_img'][:, :, off_y:off_y+min_h, off_x:off_x+min_w]
        # batch_input_params[idx, 0] = item['input_param']
        batch_input_params[idx, 0] = item['input_type']

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







name = 'train_4_restoration_operator_model'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
epoches = 130
batchsize = 12
num_workers = 12

edgeCompute = EdgeComputation()

with open('/mnt/codes/imageOperator_codes/train_4_restoration_operator_model.py', 'r') as fin:
    print(fin.read())
print("***********************************************")

##### filter_input_folder contains the orignial natural images in the dataset #####
filter_input_folder = '/mnt/data/VOC2012_L0smooth_input/'
##### the following folders contains the corrupted images taken as input in the restoration tasks #####
rain_input_folder = '/mnt/data/sky_rainy_dataset/ground_truth_even/'
sr_input_folder = '/mnt/data/VOC2012_srgray234_input_various_uniform_hw/'
noise_input_folder = '/mnt/data/VOC2012_denoisegray_input_various_uniform_hw/'
deblock_input_folder = '/mnt/data/VOC2012_deblock_input_various_uniform_hw/'

train_input_paths = []
train_label_paths = []
train_params = []
train_type = []
train_hws = []
with open("/mnt/data/VOC2012_10_operator_training_list.txt","r") as f:
    for line in f:
        label_path = line.strip()
        label_imgname = os.path.splitext(os.path.basename(label_path))[0]
        name_parts = label_imgname.split('_')
        img_h, img_w = int(name_parts[-2]), int(name_parts[-1])        
        type = label_path.split('/')[3].split('_')[1]
        # if type == 'L0smooth':
        #     param = float(name_parts[-3])
        #     train_type.append(float('0.1'))
        #     train_params.append(param)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     train_label_paths.append(label_path)
        # elif type == 'WLS':
        #     param = float(name_parts[-3])
        #     train_type.append(float('0.2'))
        #     train_params.append(param/50)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     train_label_paths.append(label_path)
        # elif type == 'RTV':
        #     param = float(name_parts[-3])
        #     train_type.append(float('0.3'))
        #     train_params.append((param-0.002)*4.125+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     train_label_paths.append(label_path)
        # elif type == 'RGF':
        #     param = float(name_parts[-3])
        #     train_type.append(float('0.4'))
        #     train_params.append((param-1)*0.022+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     train_label_paths.append(label_path)
        # elif type == 'WMF':
        #     param = float(name_parts[-3])
        #     train_type.append(float('0.5'))
        #     train_params.append((param-1)*0.022+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     train_label_paths.append(label_path)
        # elif type == 'shock':
        #     param = float(name_parts[-3])
        #     train_type.append(float('0.6'))
        #     train_params.append(0.02)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     train_label_paths.append(label_path)
        if type == 'deblock':
            param = float(name_parts[-3])
            train_type.append(float('0.1'))
            train_params.append(0.02)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            train_input_paths.append(label_path)
            train_label_paths.append(os.path.join(deblock_input_folder, base_imgname+'.png'))
        elif type == 'rainy':
            train_type.append(float('0.2'))
            train_params.append(0.02)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            train_input_paths.append(label_path)
            train_label_paths.append(os.path.join(rain_input_folder, base_imgname+'.png'))
        elif type == 'srgray234':
            param = float(name_parts[-3])
            train_type.append(float('0.3'))
            train_params.append(0.02)
            base_imgname = label_imgname.replace('_%s_%s'%(name_parts[-2], name_parts[-1]), '')
            train_label_paths.append(os.path.join(sr_input_folder, base_imgname+'.png'))
            train_input_paths.append(label_path)
        elif type == 'denoisegray':
            param = float(name_parts[-3])
            train_type.append(float('0.4'))
            train_params.append(0.02)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            train_label_paths.append(os.path.join(noise_input_folder, base_imgname+'.png'))
            train_input_paths.append(label_path)
        else:
            continue
        
        train_hws.append((img_h, img_w))

test_input_paths = []
test_label_paths = []
test_params = []
test_type = []
test_hws = []
with open("/mnt/data/VOC2012_10_operator_test_list.txt","r") as f:
    for line in f:
        label_path = line.strip()
        label_imgname = os.path.splitext(os.path.basename(label_path))[0]
        name_parts = label_imgname.split('_')
        img_h, img_w = int(name_parts[-2]), int(name_parts[-1])
        type = (label_path.split('/')[3].split('_')[1])
        # if type == 'L0smooth':
        #     param = float(name_parts[-3])
        #     test_type.append(float('0.1'))
        #     test_params.append(param)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     test_label_paths.append(label_path)
        # elif type == 'WLS':
        #     param = float(name_parts[-3])
        #     test_type.append(float('0.2'))
        #     test_params.append(param/50)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     test_label_paths.append(label_path)
        # elif type == 'RTV':
        #     param = float(name_parts[-3])
        #     test_type.append(float('0.3'))
        #     test_params.append((param-0.002)*4.125+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     test_label_paths.append(label_path)
        # elif type == 'RGF':
        #     param = float(name_parts[-3])
        #     test_type.append(float('0.4'))
        #     test_params.append((param-1)*0.022+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     test_label_paths.append(label_path)
        # elif type == 'WMF':
        #     param = float(name_parts[-3])
        #     test_type.append(float('0.5'))
        #     test_params.append((param-1)*0.022+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     test_label_paths.append(label_path)
        # elif type == 'shock':
        #     param = float(name_parts[-3])
        #     test_type.append(float('0.6'))
        #     test_params.append(0.02)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     test_label_paths.append(label_path)
        if type == 'deblock':
            param = float(name_parts[-3])
            test_type.append(float('0.1'))
            test_params.append(0.02)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            test_input_paths.append(label_path)
            test_label_paths.append(os.path.join(deblock_input_folder, base_imgname+'.png'))
        elif type == 'rainy':
            test_type.append(float('0.2'))
            test_params.append(0.02)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            test_input_paths.append(label_path)
            test_label_paths.append(os.path.join(rain_input_folder, base_imgname+'.png'))
        elif type == 'srgray234':
            param = float(name_parts[-3])
            test_type.append(float('0.3'))
            test_params.append(0.02)
            base_imgname = label_imgname.replace('_%s_%s'%(name_parts[-2], name_parts[-1]), '')
            test_label_paths.append(os.path.join(sr_input_folder, base_imgname+'.png'))
            test_input_paths.append(label_path)
        elif type == 'denoisegray':
            param = float(name_parts[-3])
            test_type.append(float('0.4'))
            test_params.append(0.02)
            base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
            test_label_paths.append(os.path.join(noise_input_folder, base_imgname+'.png'))
            test_input_paths.append(label_path)
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
    train_loss_all = [0,0,0,0]
    test_loss_all = [0,0,0,0]
    train_count_all = [0.001,0.001,0.001,0.001]
    test_count_all = [0.001,0.001,0.001,0.001]
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
    print('Epoch: %d train loss: deblock: %.4f, rainy: %.4f, superresolution: %.4f, denoise: %.4f '%(p+1,train_loss_all[0]/train_count_all[0],train_loss_all[1]/train_count_all[1],train_loss_all[2]/train_count_all[2],train_loss_all[3]/train_count_all[3]))
    print('Epoch: %d train loss: deblock: %.4f, rainy: %.4f, superresolution: %.4f, denoise: %.4f '%(p+1,test_loss_all[0]/test_count_all[0],test_loss_all[1]/test_count_all[1],test_loss_all[2]/test_count_all[2],test_loss_all[3]/test_count_all[3]))
    print('Epoch: %d average train loss: %.4f'%(p+1,train_loss))
    print('Epoch: %d average test  loss: %.4f'%(p+1,test_loss))
    print('Epoch: %d time elapsed: %.2f hours'%(p+1,elapsed/3600))
    torch.save(model.module.state_dict(),'/mnt/codes/imageOperator/netfiles/model_{}_{}.net'.format(name,p+1))
total_elapsed = time.time() - total_start_time
print('Total time elapsed: %.2f days'%(total_elapsed/(3600*24)))