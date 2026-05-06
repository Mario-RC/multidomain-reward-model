#!/bin/bash

##########################################
### stage 1 prepare ###
CUDA_VISIBLE_DEVICES=0 python3 stage-1_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path data/dataset/Multi-Domain-Data-Scoring \
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
  --dataset_path data/dataset/Multi-Domain-Data-Preference-Pairs \
  --output_dataset_name Multi-Domain-Data-Preference-Pairs \
  --dataset_split train \
  --n_shards 1 --shard_idx 1 --device 0

### stage 2 prepare for reference data ###
CUDA_VISIBLE_DEVICES=0 python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path RLHFlow/UltraFeedback-preference-standard \
  --output_dataset_name UltraFeedback-preference-standard \
  --dataset_split train \
  --n_shards 1 --shard_idx 1 --device 0

### stage 2 prepare for reward-bench eval data ###
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
  --debiasing_dims 18 20 22 \
  --temperature 2.0 \
  --n_steps 30000 \
  --seed 0 \
  --n_hidden 1 \
  --hidden_size 64 \
  --learning_rate 0.0005 \
  --weight_decay 0.0 \
  --dropout 0.1 \
  --batch_size 2048 \
  --corr_threshold 0.04 \
  --logit_scale 2.0 \
  --eval_every 200 \
  --patience 15 \
  --curriculum \
  --curriculum_phase1_frac 0.20 \
  --curriculum_phase2_frac 0.50 \
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
  --temperature 2.0 \
  --n_steps 30000 \
  --seed 0 \
  --n_hidden 1 \
  --hidden_size 64 \
  --learning_rate 0.0005 \
  --weight_decay 0.0 \
  --dropout 0.1 \
  --batch_size 2048 \
  --corr_threshold 0.04 \
  --logit_scale 2.0 \
  --curriculum \
  --output_model_name multi-domain-rm-llama-3-8b-it

##########################################
### evaluate ###
CUDA_VISIBLE_DEVICES=0 python3 evaluate.py \
  --model_name multi-domain-rm-llama-3-8b-it \
  --eval data/test

##########################################
### predict ###
CUDA_VISIBLE_DEVICES=0 python3 predict.py \
  --model_name multi-domain-rm-llama-3-8b-it

##########################################
### analyze attribute correlations ###
CUDA_VISIBLE_DEVICES=0 python3 analyze_correlations.py \
  --dataset_path data/dataset/Multi-Domain-Data-Scoring.jsonl \
  --threshold 0.5

##########################################
### evaluate baseline ###
CUDA_VISIBLE_DEVICES=0 python3 evaluate_baseline.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --eval data/test \
  --model_name multi-domain-rm-llama-3-8b-it

##########################################
### compare models ###
python3 compare_models.py \
  --model_parent_dir model \
  --no_baselines \
  --models multi-domain-rm-llama-3-8b-it multi-domain-rm-gemma-2-9b-it multi-domain-rm-qwen-3-8b-it
