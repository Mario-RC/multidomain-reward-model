#!/bin/bash

# sfairXC/FsfairX-LLaMA3-RM-v0.1

##########################################
### stage 1 prepare ###
# CUDA_VISIBLE_DEVICES=0 python3 stage-1_prepare.py \
# --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
# --dataset_path datasets/stage_1.jsonl \
# --output_dataset_name mdo \
# --n_shards 1 --shard_idx 1 --device 0

##########################################
### stage 1 train ###
# CUDA_VISIBLE_DEVICES=0 python3 stage-1_train.py \
# --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
# --dataset_name mdo

##########################################
### stage 2 prepare for preference data ###
# CUDA_VISIBLE_DEVICES=0 python3 stage-2_prepare.py \
# --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
# --model_family llama3 \
# --dataset_path datasets/stage_2.jsonl \
# --dataset_split train --n_shards 1 --shard_idx 1 --device 0

### stage 2 prepare for reference data ###
CUDA_VISIBLE_DEVICES=0 python3 stage-2_prepare.py \
--model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
--model_family llama3 \
--dataset_path RLHFlow/UltraFeedback-preference-standard \
--dataset_split train --n_shards 1 --shard_idx 1 --device 0

### stage 2 prepare for reward-bench eval data ###
# CUDA_VISIBLE_DEVICES=0 python3 stage-2_prepare.py \
# --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
# --model_family llama3 \
# --dataset_path allenai/reward-bench \
# --dataset_split filtered --n_shards 1 --shard_idx 1 --device 0

##########################################
### stage 2 train ###
# CUDA_VISIBLE_DEVICES=0 python3 stage-2_train.py --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 --model_family llama3 \
# --multi_objective_dataset mdo \
# --preference_dataset datasets/stage_2.jsonl \
# --reference_dataset RLHFlow/UltraFeedback-preference-standard \
# --eval_reward_bench --device 0

# multi-gpu stage 2 train
# torchrun --standalone --nproc_per_node=2 stage-2_train.py \
# --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
# --model_family llama3 \
# --multi_objective_dataset mdo \
# --preference_dataset datasets/stage_2.jsonl \
# --reference_dataset RLHFlow/UltraFeedback-preference-standard \
# --eval_reward_bench