#%%
import json
import pandas as pd
import os
import glob
from subprocess import call

def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(data.ix[i, 1])
    return labels


def convert_csv_to_dict(csv_path, subset, labels):
    try:
        data = pd.read_csv(csv_path, delimiter=' ', header=None)
    except pd.errors.EmptyDataError:
        return {}

    keys = []
    key_labels = []
    for i in range(data.shape[0]):
        row = data.ix[i, :]
        class_name = labels[row[1] - 1]
        basename = str(row[0])
        
        keys.append(basename)
        key_labels.append(class_name)
        
    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        label = key_labels[i]
        database[key]['annotations'] = {'label': label}
    
    return database

def convert_jester_csv_to_activitynet_json(label_csv_path, train_csv_path, test_csv_path, dst_json_path):
    labels = load_labels(label_csv_path)
    if train_csv_path:
        train_database = convert_csv_to_dict(train_csv_path, 'training', labels)
    else:
        train_database = {}
    test_database = convert_csv_to_dict(test_csv_path, 'testing', labels)
    
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(test_database)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)

#%%
def prepare_json(csv_dir_path='./annotation_ems', expr_name='15.3'):
    """ Convert training & testing list into a single json file.

    Parameters
    ----------
        csv_dir_path: the path to training/testing list; the output json file will also be saved into the same directory.
        expr_name: expriment name.
    """
    label_csv_path = os.path.join(csv_dir_path, 'classInd%s.txt' % expr_name)
    train_csv_path = os.path.join(csv_dir_path, 'trainlist%s.txt' % expr_name)
    test_csv_path = os.path.join(csv_dir_path, 'testlist%s.txt' % expr_name)
    dst_json_path = os.path.join(csv_dir_path, 'ems%s.json' % expr_name)

    convert_jester_csv_to_activitynet_json(
        label_csv_path, train_csv_path, test_csv_path, dst_json_path)

def split(video_path, annot_path, fps=30, delay=4/30, duration=10/30):
    """ Split a single video file into multiple clips based on annotation.

    Parameters
    ----------
        fps: frame rate
        delay: how many seconds it take for bootstrap
        duration: the length for each clip, in second(s)

    """
    directory = video_path.split(".")[0] + "_all"
    if not os.path.exists(directory):
        os.makedirs(directory)
        call(["ffmpeg", "-i",  video_path, os.path.join(directory, "%05d.jpg"), "-hide_banner"])

    with open(annot_path, 'r') as f:
        annot = f.readlines()

    annot = [a for a in annot[0::2]]
    ges_cnt = {}

    for j, a in enumerate(annot[:]):
        ges = a.split('start')[0]
        ges = '_'.join(ges.lower().strip().split(' '))
        # ges = 'human_' + ges

        t = a.split('start:')[-1].strip()
        t = float(t)

        start = int((t + delay) * fps)
        end = int((t + delay + duration) * fps)

        cnt = ges_cnt.get(ges, 0) + 1
        ges_cnt[ges] = cnt
        output_dir = os.path.join('/'.join(video_path.split('/')[:-1]), '{:03d}_{}_{:02d}_all'.format(j, ges, cnt))
        os.makedirs(output_dir, exist_ok=True)
        cmd = 'cp '
        for i in range(start, end):
            cmd += '{}/{:05d}.jpg '.format(directory, i)
        os.system(cmd + ' %s' % output_dir)