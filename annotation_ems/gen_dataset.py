import os
import glob
import math
import random

### begin of config

random.seed(666)

dataset_path = '/fastdata/yxchen/gesture-datasets/ems'
output_path = './annotation_ems'

round = "14.2"
modality = "rgb" # d, rgb, rgbd

# train: first n
train_partition = {
    'subject01_gestures2_base': 20,
}

# test: all except first n
test_partition = {
    'subject01_gestures2_base': 20
}

labels = ['wrist_up', 'wrist_down', 'wrist_left', 'wrist_right',
        'arm_down', 'up_left', 'up_right', 'down_left', 
        'arm_down_left', 'arm_down_right']

# labels_human = ['human_' + l for l in labels]
# labels += labels_human

### end of config


def get_list(path, dspath, paired=False, modality='rgb'):
    if modality == 'd':
        samples = sorted(glob.glob(os.path.join(path, 'depth/*_all')))
    elif modality == 'rgb':
        samples = sorted(glob.glob(os.path.join(path, 'rgb/*_all')))
    elif modality == 'rgbd':
        samples = sorted(glob.glob(os.path.join(path, 'rgb/*_all'))) + \
            sorted(glob.glob(os.path.join(path, 'depth/*_all')))

    random.shuffle(samples)
    l = {}
    for s in samples:
        s = os.path.relpath(s, dspath)
        label = get_label_id(s, paired=paired)
        if label != None:
            l[label] = l.get(label, [])
            l[label].append(s)
    return l

def make_dataset(path, paired=False, modality='rgb'):
    dataset = {}
    dpath = os.path.join(path, 'data')
    for d in glob.glob(os.path.join(dpath, '*')):
        dataset[os.path.relpath(d, dpath)] = get_list(os.path.join(dpath, d), path, paired=paired, modality=modality)
    
    return dataset

def get_label_id(path, paired=False):
    if paired:
        path = path.split('FOLLOWED_BY')[-1]
    for i, l in enumerate(labels):
        if l in path:
            return i
    return None

def gen_list(dataset, partition, labels, stage='train'):
    l = []
    for k in partition.keys():
        data = dataset[k]
        part = partition[k]
        for i, label in enumerate(labels):
            if not i in data.keys():
                continue
            if stage == 'train':
                l += [(x, str(i+1)) for x in data[i][:part]]
            elif stage == 'test':
                l += [(x, str(i+1)) for x in data[i][part if part!=None else len(data[i]):]]
            else:
                raise NotImplementedError()
    return l

def write_list(l, path):
    with open(path, 'w') as f:
        f.write('\n'.join([' '.join(x) for x in l]))

def write_labels(labels, path):
    class_ind = [' '.join((str(i+1), x)) for i, x in enumerate(labels)]
    with open(path, 'w') as f:
        f.write('\n'.join(class_ind))

labels.sort(key=lambda item: (-len(item), item))

dataset = make_dataset(dataset_path, paired=True, modality=modality)
train_list = gen_list(dataset, train_partition, labels, 'train')
test_list = gen_list(dataset, test_partition, labels, 'test')
write_list(train_list, os.path.join(output_path, 'trainlist' + round + '.txt'))
write_list(test_list, os.path.join(output_path, 'testlist' + round + '.txt'))
write_labels(labels, os.path.join(output_path, 'classInd' + round + '.txt'))
