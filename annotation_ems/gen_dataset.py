import os
import glob
import math

dataset_path = '/mnt/data/yxchen/gesture-datasets/ems'
output_path = './annotation_ems'

round = "05.5"

# train: first n
train_partition = {
    'subject01': 0,
    'subject01_new_cloth_reattached': 20,
    'subject01_diff_bg': 0
}

# test: all except first n
test_partition = {
    'subject01': 0,
    'subject01_new_cloth_reattached': 20,
    'subject01_diff_bg': None 
}

labels = ['wrist_up', 'wrist_down', 'wrist_left', 'wrist_right']

def get_list(path, dspath):
    samples = sorted(glob.glob(os.path.join(path, '*_all')))
    l = {}
    for s in samples:
        s = os.path.relpath(s, dspath)
        label = get_label_id(s)
        if label != None:
            l[label] = l.get(label, [])
            l[label].append(s)
    return l

def make_dataset(path):
    dataset = {}
    dpath = os.path.join(path, 'data')
    for d in glob.glob(os.path.join(dpath, '*')):
        dataset[os.path.relpath(d, dpath)] = get_list(os.path.join(dpath, d), path)
    
    return dataset

def get_label_id(path):
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

dataset = make_dataset(dataset_path)
train_list = gen_list(dataset, train_partition, labels, 'train')
test_list = gen_list(dataset, test_partition, labels, 'test')
write_list(train_list, os.path.join(output_path, 'trainlist' + round + '.txt'))
write_list(test_list, os.path.join(output_path, 'testlist' + round + '.txt'))
write_labels(labels, os.path.join(output_path, 'classInd' + round + '.txt'))