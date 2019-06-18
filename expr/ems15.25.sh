#!/bin/bash

# train classifier
CUDA_VISIBLE_DEVICES=0 python main.py \
	--root_path /home/yxchen/ems-gesture/Real-time-GesRec \
	--video_path /fastdata/yxchen/gesture-datasets/ems \
	--annotation_path annotation_ems/ems15.25.json\
	--result_path results/ems15.25 \
	--pretrain_path results/ems15.12/save_30.pth \
	--dataset ems \
	--sample_duration 32 \
    --learning_rate 0.01 \
    --model resnext \
	--model_depth 101 \
	--resnet_shortcut B \
	--batch_size 48 \
	--n_classes 4 \
	--n_finetune_classes 4 \
	--n_threads 16 \
	--checkpoint 1 \
	--modality RGB \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test \
    --n_epochs 10 \
    --no_val \
    --checkpoint 5
