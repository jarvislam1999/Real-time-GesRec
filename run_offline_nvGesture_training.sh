#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2
train_classifier=false

if $train_classifier; then
# train classifier
python main.py \
	--root_path /home/yxchen/Real-time-GesRec \
	--video_path /mnt/data/yxchen/gesture-datasets/nvGesture \
	--annotation_path annotation_nvGesture/nvall.json\
	--result_path results/nv_color_classifier \
	--resume_path /mnt/data/yxchen/model-zoo/jester_resnext_101_RGB_32.pth \
	--dataset nv \
	--sample_duration 32 \
    --learning_rate 0.01 \
    --model resnext \
	--model_depth 101 \
	--resnet_shortcut B \
	--batch_size 8 \
	--n_classes 26 \
	--n_finetune_classes 27 \
	--n_threads 16 \
	--checkpoint 1 \
	--modality RGB \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test \
     --n_epochs 100 \

else
# train detector
python main.py \
	--root_path /home/yxchen/Real-time-GesRec \
	--video_path /mnt/data/yxchen/gesture-datasets/nvGesture \
	--annotation_path annotation_nvGesture/nvbinary.json\
	--result_path results/nv_color_detector \
	--resume_path /home/yxchen/Real-time-GesRec/results/nv_color_detector/model_checkpoint.pth \
	--dataset nv \
	--sample_duration 8 \
    --learning_rate 0.01 \
    --model resnetl \
	--model_depth 10 \
	--resnet_shortcut A \
	--batch_size 8 \
	--n_classes 2 \
	--n_finetune_classes 2 \
	--n_threads 16 \
	--checkpoint 1 \
	--modality RGB \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test \
     --n_epochs 500 \
	 --weighted

fi