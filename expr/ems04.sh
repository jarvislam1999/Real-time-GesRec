#!/bin/bash


# train classifier
CUDA_VISIBLE_DEVICES=1 python main.py \
	--root_path /home/yxchen/Real-time-GesRec \
	--video_path /mnt/data/yxchen/gesture-datasets/ems \
	--annotation_path annotation_ems/ems04.json\
	--result_path results/ems04 \
	--pretrain_path /mnt/data/yxchen/model-zoo/jester_resnext_101_RGB_32.pth \
	--dataset ems \
	--sample_duration 32 \
    --learning_rate 0.01 \
    --model resnext \
	--model_depth 101 \
	--resnet_shortcut B \
	--batch_size 8 \
	--n_classes 27 \
	--n_finetune_classes 4 \
	--n_threads 16 \
	--checkpoint 1 \
	--modality RGB \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test \
    --n_epochs 100 \
