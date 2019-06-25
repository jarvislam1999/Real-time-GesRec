import sys
sys.path.append('.')

from annotation_ems.gen_dataset import generate_dataset
from annotation_ems.ems_prepare import split, prepare_json

### begin of config

dataset_path = '/mnt/data/jarvislam1999'
output_path = './annotation_ems'

expr_name = "16.8"
modality = "rgb"  # d, rgb, rgbd

# train: first n
train_partition = {
    'subject01_nobg': 40,
    'subject01_bg02': 40,
    'subject01_bg03': 40,
    'subject01_bg04': 40,
    'subject01_bg05': 40,
    'subject01_bg06': 40
}

# test: all except first n
test_partition = {
    'subject01_nobg': 40,
    'subject01_bg01': 0,
    'subject01_bg02': 40,
    'subject01_bg03': 40,
    'subject01_bg04': 40,
    'subject01_bg05': 40,
    'subject01_bg06': 40
}

labels = ['wrist_left', 'wrist_right', 'pronation', 'supination']

### end of config

generate_dataset(expr_name=expr_name, modality=modality, dataset_path=dataset_path, output_path=output_path, train_partition=train_partition, test_partition=test_partition, labels=labels)

prepare_json(csv_dir_path=output_path, expr_name=expr_name)
