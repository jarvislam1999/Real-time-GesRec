import torch
import torch.utils.data as data
from PIL import Image
from spatial_transforms import *
import os
import math
import functools
import json
import copy
from numpy.random import randint
import numpy as np
import random

from utils import load_value_file
import pdb


def pil_loader(path, modality):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        #print(path)
        with Image.open(f) as img:
            if modality == 'RGB':
                return img.convert('RGB')
            elif modality == 'Depth':
                return img.convert('L') # 8-bit pixels, black and white check from https://pillow.readthedocs.io/en/3.0.x/handbook/concepts.html


def accimage_loader(path, modality):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, modality, sample_duration, image_loader):
    
    video = []
    """
    vid_duration = len(frame_indices)
    rand_end = max(0, vid_duration - 32 - 1)
    begin_index = random.randint(0, rand_end)
    end_index = min(begin_index + 32, vid_duration)
    out = frame_indices[begin_index:end_index]
 
    for index in out:
        if len(out) >= 32:
            break
        out.append(index)
    """
    #inner_shift = randint(3)
    #inner_shift = 0
    #frame_indices = [out[i+inner_shift+1+3] for i in range(0, 32-1, 8)]
    #frame_indices = out
    #print(frame_indices)
    if modality == 'RGB':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
            if os.path.exists(image_path):
                
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'Depth':

        for i in frame_indices:
            image_path = os.path.join(video_dir_path.replace('color','depth'), '{:05d}.jpg'.format(i) )
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'RGB-D':
        for i in frame_indices: # index 35 is used to change img to flow
            image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))

            
            image_path_depth = os.path.join(video_dir_path.replace('color','depth'), '{:05d}.jpg'.format(i) )

            
            image = image_loader(image_path, 'RGB')
            image_depth = image_loader(image_path_depth, 'Depth')

            #r,g,b = image.split()
            #d, = image_depth.split()

            #stacked_image = Image.merge ('RGBA', (r,g,b,d))


            # np_image_depth = np.array(image_depth)
            # np_image_depth = np.reshape(np_image_depth, np_image_depth.shape + (1,))

            # stacked_image = np.concatenate([np.array(image), np_image_depth], axis = 2)

            if os.path.exists(image_path):
                video.append(image)
                video.append(image_depth)
            else:
                return video
    elif modality == 'RGBDiff':
        for i in frame_indices: # index 35 is used to change img to flow
            image_path_first  = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
            image_path_second = os.path.join(video_dir_path, '{:06d}.jpg'.format(i-1))

            if not os.path.exists(image_path_second):
                image_path_second = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))

            image_first  = image_loader(image_path_first, 'RGB')
            image_second = image_loader(image_path_second, 'RGB')

            if os.path.exists(image_path_first):
                video.append(image_first)
                video.append( Image.fromarray(np.asarray(image_first) - np.asarray(image_second)) ) 
            else:
                return video    
    elif modality == 'RGBStack':
        for i in frame_indices: # index 35 is used to change img to flow
            image_path_first  = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
            image_path_second = os.path.join(video_dir_path, '{:06d}.jpg'.format(max(1,i-4)))
            #image_path_third = os.path.join(video_dir_path, '{:06d}.jpg'.format(max(1,i-2)))
            #image_path_fourth = os.path.join(video_dir_path, '{:06d}.jpg'.format(max(1,i-3)))

            image_first  = image_loader(image_path_first, 'RGB')
            image_second = image_loader(image_path_second, 'RGB')
            #image_third  = image_loader(image_path_third, 'RGB')
            #image_fourth = image_loader(image_path_fourth, 'RGB')


            if os.path.exists(image_path_first):
                video.append(image_first)
                video.append(image_second)
                #video.append(image_third)
                #video.append(image_fourth)

            else:
                return video
    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            #video_names.append('{}/{}'.format(label, key))
            video_names.append(key.split('^')[0])
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    print("[INFO]: NV Dataset - " + subset + " is loading...")
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        
        if not os.path.exists(video_path):
            continue

        

        begin_t = int(annotations[i]['start_frame'])
        end_t = int(annotations[i]['end_frame'])
        n_frames = end_t - begin_t + 1
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            #'video_id': video_names[i].split('/')[1]
            'video_id': i
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(begin_t, end_t + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class

# root_path = opt.video_path
# annotation_path = opt.annotation_path
# subset = 'training'
# spatial_transform=spatial_transform
# temporal_transform=temporal_transform
# target_transform=target_transform
# sample_duration=opt.sample_duration
# modality='RGB'


class NV(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 modality='RGB',
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.modality = modality
        self.sample_duration = sample_duration
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']


        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices, self.modality, self.sample_duration)
        oversample_clip =[]
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
    
        im_dim = clip[0].size()[-2:]
        clip = torch.cat(clip, 0).view((self.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
        
     
        #clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        #return oversample_clip, target
        return clip, target

    def __len__(self):
        return len(self.data)

