#!/bin/bash

##########################################
### stage 1 prepare ###
CUDA_VISIBLE_DEVICES=0 python3 stage-1_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path data/Multi-Domain-Data-Scoring \
  --output_dataset_name Multi-Domain-Data-Scoring \
  --dataset_split train \
  --n_shards 1 --shard_idx 1 --device 0

##########################################
### stage 1 train ###
CUDA_VISIBLE_DEVICES=0 python3 stage-1_train.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --multi_objective_dataset_name Multi-Domain-Data-Scoring \
  --dataset_split train

##########################################
### stage 2 prepare for preference data ###
CUDA_VISIBLE_DEVICES=0 python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path data/Multi-Domain-Data-Preference-Pairs \
  --output_dataset_name Multi-Domain-Data-Preference-Pairs \
  --dataset_split train \
  --n_shards 1 --shard_idx 1 --device 0

## stage 2 prepare for reference data ###
CUDA_VISIBLE_DEVICES=0 python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path RLHFlow/UltraFeedback-preference-standard \
  --output_dataset_name UltraFeedback-preference-standard \
  --dataset_split train \
  --n_shards 1 --shard_idx 1 --device 0

## stage 2 prepare for reward-bench eval data ###
CUDA_VISIBLE_DEVICES=0 python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path allenai/reward-bench \
  --output_dataset_name reward-bench \
  --dataset_split filtered \
  --n_shards 1 --shard_idx 1 --device 0

##########################################
### stage 2 train ###
CUDA_VISIBLE_DEVICES=0 python3 stage-2_train.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --multi_objective_dataset_name Multi-Domain-Data-Scoring \
  --preference_dataset_name Multi-Domain-Data-Preference-Pairs \
  --reference_dataset_name null \
  --debiasing_dim -1 \
  --dataset_split train \
  --eval reward-bench \
  --device 0

##########################################
### stage 3 packaging model ###
CUDA_VISIBLE_DEVICES=0 python3 stage-3_package_model.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --multi_objective_dataset_name Multi-Domain-Data-Scoring \
  --preference_dataset_name Multi-Domain-Data-Preference-Pairs \
  --reference_dataset_name null \
  --output_model_name multi-domain-rm-llama-3-8b-it

##########################################
### evaluate ###
CUDA_VISIBLE_DEVICES=0 python3 evaluate.py \
  --model_name multi-domain-rm-llama-3-8b-it

##########################################
### predict ###
CUDA_VISIBLE_DEVICES=0 python3 predict.py \
  --model_name multi-domain-rm-llama-3-8b-it

##########################################
### analyze attribute correlations ###
CUDA_VISIBLE_DEVICES=0 python3 analyze_correlations.py \
  --dataset_path data/Multi-Domain-Data-Scoring.jsonl \
  --threshold 0.5

##########################################
### evaluate baseline (no regression) ###
CUDA_VISIBLE_DEVICES=0 python3 evaluate_baseline.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --no_regression \
  --model_name multi-domain-rm-llama-3-8b-it

##########################################
### benchmark ###
python3 benchmark.py \
  --model_parent_dir model
