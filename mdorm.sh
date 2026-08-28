#!/bin/bash

set -euo pipefail

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
  --output_dataset_name Multi-Domain-Data-Preference-Pairs-SharedGate \
  --prompt_batch_size 8 \
  --dataset_split train \
  --n_shards 1 --shard_idx 1 --device 0

# Optional reference/RewardBench preparation is omitted by default.
# It is unnecessary when reference_dataset_name=null and eval=null.

##########################################
### stage 2 train ###
CUDA_VISIBLE_DEVICES=0 python3 stage-2_train.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --multi_objective_dataset_name Multi-Domain-Data-Scoring \
  --preference_dataset_name Multi-Domain-Data-Preference-Pairs-SharedGate \
  --reference_dataset_name null \
  --debiasing_dims -1 \
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
  --domain_loss_weight 0.25 \
  --entropy_weight 0.02 \
  --load_balance_weight 0.05 \
  --eval_every 200 \
  --patience 15 \
  --curriculum \
  --curriculum_phase1_frac 0.20 \
  --curriculum_phase2_frac 0.50 \
  --dataset_split train \
  --device 0

##########################################
### stage 3 packaging model ###
CUDA_VISIBLE_DEVICES=0 python3 stage-3_package_model.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --multi_objective_dataset_name Multi-Domain-Data-Scoring \
  --preference_dataset_name Multi-Domain-Data-Preference-Pairs-SharedGate \
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
  --domain_loss_weight 0.25 \
  --entropy_weight 0.02 \
  --load_balance_weight 0.05 \
  --curriculum \
  --output_model_name multi-domain-rm-fsfairx-llama-3-8b-it

##########################################
### evaluate ###
CUDA_VISIBLE_DEVICES=0 python3 evaluate.py \
  --model_name multi-domain-rm-fsfairx-llama-3-8b-it \
  --eval data/test

##########################################
### predict ###
CUDA_VISIBLE_DEVICES=0 python3 predict.py \
  --model_name multi-domain-rm-fsfairx-llama-3-8b-it

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
  --model_name multi-domain-rm-fsfairx-llama-3-8b-it

##########################################
### compare models ###
python3 compare_models.py \
  --model_parent_dir model \
  --no_baselines \
  --models \
    multi-domain-rm-fsfairx-llama-3-8b-it \
    multi-domain-rm-fsfairx-gemma-2-9b-it \
    multi-domain-rm-qwen-3-nemotron-8b-it \
    multi-domain-rm-mistral-7b-it \
    multi-domain-rm-skywork-llama-3.1-8b-it \
    multi-domain-rm-skywork-qwen-3-8b-it
