import test
from validation import val_epoch
from train import train_epoch
from utils import AverageMeter, calculate_precision, calculate_recall
from utils import Logger
from dataset import get_training_set, get_validation_set, get_test_set, get_online_data
from target_transforms import Compose as TargetCompose
from target_transforms import ClassLabel, VideoID
from temporal_transforms import *
from spatial_transforms import *
from mean import get_mean, get_std
from model import generate_model
from opts import parse_opts_offline
from torch.nn import functional as F
from torch.autograd import Variable
import torch
import torchvision
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import shutil
import argparse
import math
import pandas as pd
import json
import sys
import time
import os
import random
import warnings

class EMSTester():
    def __init__(self, root_path='', video_path='', annotation_path='', result_path='', model_path=''):

        opt = parse_opts_offline(
            ['--root_path', root_path,
            '--video_path', video_path, 
            '--annotation_path', annotation_path,
            '--result_path', result_path,
            '--resume_path', model_path,
            '--dataset', 'ems',
            '--sample_duration', '32',
            '--model', 'resnext',
            '--model_depth', '101',
            '--resnet_shortcut', 'B',
            '--batch_size', '1',
            '--n_finetune_classes', '4',
            '--n_threads', '1',
            '--checkpoint', '1',
            '--modality', 'RGB',
            '--n_val_samples', '1',
            '--test_subset', 'test']
        )

        if opt.root_path != '':
            opt.video_path = os.path.join(opt.root_path, opt.video_path)
            opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
            opt.result_path = os.path.join(opt.root_path, opt.result_path)
            if opt.resume_path:
                opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
            if opt.pretrain_path:
                opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
        opt.scales = [opt.initial_scale]
        for i in range(1, opt.n_scales):
            opt.scales.append(opt.scales[-1] * opt.scale_step)
        opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
        opt.mean = get_mean(opt.norm_value)
        opt.std = get_std(opt.norm_value)

        print(opt)

        #%%
        warnings.filterwarnings('ignore')

        torch.manual_seed(opt.manual_seed)

        model, parameters = generate_model(opt)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                                p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)

        self.opt = opt
        self.model = model
        self.parameters = parameters

    def test(self, annotation_path='', video_path=''):
        opt = self.opt
        
        if annotation_path != '':
            opt.annotation_path = annotation_path
            if opt.root_path != '':
                opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        
        if video_path != '':
            opt.video_path = video_path
            if opt.root_path != '':
                opt.video_path = os.path.join(opt.root_path, opt.video_path)

        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)

        with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file)

        if opt.no_mean_norm and not opt.std_norm:
            norm_method = Normalize([0, 0, 0], [1, 1, 1])
        elif not opt.std_norm:
            norm_method = Normalize(opt.mean, [1, 1, 1])
        else:
            norm_method = Normalize(opt.mean, opt.std)

        # original
        spatial_transform = Compose([
            #Scale(opt.sample_size),
            Scale(112),
            CenterCrop(112),
            ToTensor(opt.norm_value), norm_method
        ])

        temporal_transform = TemporalCenterCrop(opt.sample_duration)


        target_transform = ClassLabel()
        # test_data = get_test_set(
            # opt, spatial_transform, temporal_transform, target_transform)
        test_data = torchvision.datasets.Fakedata(size=400, image_size=(3, 112, 112), num_classes=4)

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)

        recorder = []

        self.model.eval()

        batch_time = AverageMeter()
        end_time = time.time()

        for i, (inputs, targets) in enumerate(test_loader):
            if not opt.no_cuda:
                targets = targets.cuda(non_blocking=True)
            #inputs = Variable(torch.squeeze(inputs), volatile=True)
            with torch.no_grad():
                inputs = Variable(inputs)
                targets = Variable(targets)
                outputs = self.model(inputs)
                if not opt.no_softmax_in_test:
                    outputs = F.softmax(outputs, dim=1)
                recorder.append(outputs.data.cpu().numpy().copy())

            _cls = outputs.argmax(1).cpu().numpy().tolist()[0]

            batch_time.update(time.time() - end_time)
            end_time = time.time()

        print('-----Evaluation is finished------')
        print('Avg Time: %.5fs' % batch_time.avg)


ems_tester = EMSTester()
ems_tester.test()