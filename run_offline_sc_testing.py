#%% [markdown]
# ## To test classifier and detector trained on nvGesture

#%%
%%bash
CUDA_VISIBLE_DEVICES=1 python offline_test.py \
	--root_path /home/yxchen/Real-time-GesRec \
	--video_path /mnt/data/yxchen/gesture-datasets/self-collect \
	--annotation_path annotation_sc/sc.json\
	--result_path results/sc_test \
	--resume_path /mnt/data/yxchen/model-zoo/jester_resnext_101_RGB_32.pth \
	--dataset sc \
	--sample_duration 32 \
    --model resnext \
	--model_depth 101 \
	--resnet_shortcut B \
	--batch_size 1 \
	--n_classes 27 \
	--n_finetune_classes 27 \
	--n_threads 16 \
	--checkpoint 1 \
	--modality RGB \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test

#%%