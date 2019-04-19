#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python main.py \
	--root_path /home/yxchen/Real-time-GesRec \
	--video_path /mnt/data/yxchen/gesture-datasets/nvGesture \
	--annotation_path annotation_nvGesture/nvall.json\
	--result_path results \
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

