import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from validation import val_epoch
from train import train_epoch
from utils import AverageMeter
from mean import get_mean, get_std
from model import generate_model
from opts import parse_opts_offline
from torch.nn import functional as F
from torch.autograd import Variable
import torch
import torchvision
import itertools
import numpy as np
import shutil
import argparse
import math
import json
import sys
import time
import os
import random
import warnings


class FakeData():
    """A fake dataset that returns randomly generated images and returns them as PIL images

    Args:
        size (int, optional): Size of the dataset. Default: 1000 images
        image_size(tuple, optional): Size if the returned images. Default: (3, 224, 224)
        num_classes(int, optional): Number of classes in the datset. Default: 10
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, `transforms.RandomCrop`
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        random_offset (int): Offsets the index-based random seed used to
            generate each image. Default: 0

    """

    def __init__(self, size=1000, image_size=(3, 224, 224), num_classes=10,
                 transform=None, target_transform=None, random_offset=0):
        #super(FakeData, self).__init__(None)
        self.transform = transform
        self.target_transform = target_transform
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform
        self.target_transform = target_transform
        self.random_offset = random_offset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError("{} index out of range".format(
                self.__class__.__name__))
        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)
        img = torch.randn(*self.image_size)
        target = torch.randint(0, self.num_classes,
                               size=(1,), dtype=torch.long)[0]
        torch.set_rng_state(rng_state)

        return img, target

    def __len__(self):
        return self.size


class EMSTester():
    def __init__(self, root_path='', video_path='', annotation_path='', result_path='', model_path='', sample_duration=32, modality='RGB', sample_size=112):

        opt = parse_opts_offline(
            ['--root_path', root_path,
             '--video_path', video_path,
             '--annotation_path', annotation_path,
             '--result_path', result_path,
             '--resume_path', model_path,
             '--dataset', 'ems',
             '--sample_duration', str(sample_duration),
             '--model', 'resnext',
             '--model_depth', '101',
             '--resnet_shortcut', 'B',
             '--batch_size', '1',
             '--n_finetune_classes', '4',
             '--n_threads', '1',
             '--checkpoint', '1',
             '--modality', modality,
             '--n_val_samples', '1',
             '--sample_size', str(sample_size),
             '--test_subset', 'test']
        )
        self.sample_duration = sample_duration
        self.modality = modality
        self.sample_size = sample_size

        if opt.root_path != '':
            opt.video_path = os.path.join(opt.root_path, opt.video_path)
            opt.annotation_path = os.path.join(
                opt.root_path, opt.annotation_path)
            opt.result_path = os.path.join(opt.root_path, opt.result_path)
            if opt.resume_path:
                opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
            if opt.pretrain_path:
                opt.pretrain_path = os.path.join(
                    opt.root_path, opt.pretrain_path)
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
                opt.annotation_path = os.path.join(
                    opt.root_path, opt.annotation_path)

        if video_path != '':
            opt.video_path = video_path
            if opt.root_path != '':
                opt.video_path = os.path.join(opt.root_path, opt.video_path)

        nch = 3 if self.modality == 'RGB' else 1
        test_data = FakeData(size=1, image_size=(
            nch, self.sample_duration, self.sample_size, self.sample_size),num_classes=4)

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True)

        recorder = []

        self.model.eval()

        batch_time = AverageMeter()
        end_time = time.time()

        print('start evaluation')

        for j in range(100):
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


ems_tester = EMSTester(sample_duration=10, modality='RGB', sample_size=56)
ems_tester.test()
