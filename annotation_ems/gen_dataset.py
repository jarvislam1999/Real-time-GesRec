import os
import glob
import math
import random


def get_files_of_label(path, dspath, labels, paired=False, modality='rgb'):
    '''
        Return a dict, key is a label, value is a list of files of that label.
    '''
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
        label = get_label_id(s, labels, paired=paired)
        if label != None:
            l[label] = l.get(label, [])
            l[label].append(s)
    return l

def make_dataset(path, labels, paired=False, modality='rgb'):
    '''
        Load all dataset under a specific path.
    '''
    dataset = {}
    dpath = os.path.join(path, 'data')
    for d in glob.glob(os.path.join(dpath, '*')):
        dataset[os.path.relpath(d, dpath)] = get_files_of_label(
            os.path.join(dpath, d), path, labels, paired=paired, modality=modality)
    
    return dataset

def get_label_id(path, labels, paired=False):
    '''
        Given a file path, return its label id.
    '''
    if paired:
        path = path.split('FOLLOWED_BY')[-1]
    for i, l in enumerate(labels):
        if l in path:
            return i
    return None

def gen_list(dataset, partition, labels, stage='train', sort_by_filename=False):
    '''
        Generate a list of all files specified in `partition`.
    '''
    l = []
    for k in partition.keys():
        data = dataset[k]
        for i, label in enumerate(labels):
            if not i in data.keys():
                continue
            part = min(partition[k], len(data[i]))
            if stage == 'train':
                l += [(x, str(i+1)) for x in data[i][:part]]
            elif stage == 'test':
                l += [(x, str(i+1)) for x in data[i][part if part!=None else len(data[i]):]]
            else:
                raise NotImplementedError()
    if sort_by_filename:
        l.sort(key=lambda item: item[0])
    return l

def write_list(l, path):
    with open(path, 'w') as f:
        f.write('\n'.join([' '.join(x) for x in l]))

def write_labels(labels, path):
    class_ind = [' '.join((str(i+1), x)) for i, x in enumerate(labels)]
    with open(path, 'w') as f:
        f.write('\n'.join(class_ind))


def generate_dataset(expr_name, train_partition, test_partition, labels, modality, dataset_path, output_path='./annotation_ems', random_seed=666, sort_by_filename=False):

    random.seed(666)

    labels.sort(key=lambda item: (-len(item), item))

    dataset = make_dataset(dataset_path, labels, paired=True, modality=modality)
    train_list = gen_list(dataset, train_partition, labels,
                          'train', sort_by_filename=sort_by_filename)
    test_list = gen_list(dataset, test_partition, labels,
                         'test', sort_by_filename=sort_by_filename)
    write_list(train_list, os.path.join(
        output_path, 'trainlist' + expr_name + '.txt'))
    write_list(test_list, os.path.join(
        output_path, 'testlist' + expr_name + '.txt'))
    write_labels(labels, os.path.join(
        output_path, 'classInd' + expr_name + '.txt'))


if __name__ == '__main__':
    ### begin of config

    dataset_path = '/fastdata/yxchen/gesture-datasets/ems'
    output_path = './annotation_ems'

    round = "15.3"
    modality = "rgb"  # d, rgb, rgbd

    # train: first n
    train_partition = {
        'subject01_machine_recovery_3gps_02': 50,
    }

    # test: all except first n
    test_partition = {
        'subject01_machine_recovery_3gps_02': 50
    }

    labels = ['wrist_left', 'wrist_right', 'pronation', 'supination']

    # labels_human = ['human_' + l for l in labels]
    # labels += labels_human

    ### end of config

    generate_dataset(expr_name=round, modality=modality, dataset_path=dataset_path, output_path=output_path, train_partition=train_partition, test_partition=test_partition, labels=labels)
