#%% [markdown]
# ## To test classifier and detector trained on nvGesture

#%% 
%%capture --no-stdout
%matplotlib inline

# Test classifier
# !CUDA_VISIBLE_DEVICES=1,2,3
# !echo $CUDA_VISIBLE_DEVICES
%run offline_test.py \
	--root_path /home/yxchen/Real-time-GesRec \
	--video_path /mnt/data/yxchen/gesture-datasets/20bn-jester-v1/data \
	--annotation_path annotation_Jester/jester.json\
	--result_path results/jester_val \
	--resume_path /mnt/data/yxchen/model-zoo/jester_resnext_101_RGB_32.pth \
	--dataset jester \
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
	--test_subset val


#%%
%%bash
CUDA_VISIBLE_DEVICES=1 python offline_test.py \
	--root_path /home/yxchen/Real-time-GesRec \
	--video_path /mnt/data/yxchen/gesture-datasets/20bn-jester-v1/data \
	--annotation_path annotation_Jester/jester.json\
	--result_path results/jester_test \
	--resume_path /mnt/data/yxchen/model-zoo/jester_resnext_101_RGB_32.pth \
	--dataset jester \
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
