import sys
sys.path.append('.')

from annotation_ems.gen_dataset import generate_dataset
from annotation_ems.ems_prepare import split, prepare_json

### begin of config

dataset_path = '/mnt/data/yxchen/gesture-datasets/ems/'
output_path = './annotation_ems'

expr_name = "22.11"
modality = "rgb"  # d, rgb, rgbd

# train: first n
train_partition = {
}

# test: all except first n
test_partition = {
    'subject01_setting3_07': 0
}

labels = ['wrist_left', 'wrist_right', 'pronation', 'supination']

### end of config

generate_dataset(expr_name=expr_name, modality=modality, dataset_path=dataset_path, output_path=output_path, train_partition=train_partition, test_partition=test_partition, labels=labels)

prepare_json(csv_dir_path=output_path, expr_name=expr_name)
