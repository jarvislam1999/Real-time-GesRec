#%% [markdown]
# ## To test classifier and detector trained on nvGesture

#%% 

# Test classifier with the best model
!python offline_test.py \
	--root_path /home/yxchen/Real-time-GesRec \
	--video_path /mnt/data/yxchen/gesture-datasets/nvGesture \
	--annotation_path annotation_nvGesture/nvall_but_None.json\
	--result_path results/nv_color_classifier_test \
	--resume_path /home/yxchen/Real-time-GesRec/results/nv_color_classifier/model_checkpoint.pth \
	--dataset nv \
	--sample_duration 32 \
    --model resnext \
	--model_depth 101 \
	--resnet_shortcut B \
	--batch_size 8 \
	--n_classes 27 \
	--n_finetune_classes 27 \
	--n_threads 16 \
	--checkpoint 1 \
	--modality RGB \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test

#%%

# Test classifier with the last model